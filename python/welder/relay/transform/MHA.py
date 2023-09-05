
from typing import List, Tuple, Callable

import tvm
from tvm import relay, te, ir
from tvm.relay import reg
from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_op, is_constant, is_tuple_get_item, is_tuple, rewrite
from tvm.relay.op.contrib.register import register_pattern_table
from ..integration import add_extern_lib

"""
    To use fused multi-head attention, you need to clone and build this repo first:
        git clone https://github.com/tlc-pack/libflash_attn.git --recursive
        cd libflash_attn && mkdir build && cd build && cmake .. -DCMAKE_CXX_STANDARD=17 && make -j && cd ..
        export CPLUS_INCLUDE_PATH="$PWD/include:$CPLUS_INCLUDE_PATH"
        export LIBRARY_PATH="$PWD/build/src:$LIBRARY_PATH"
        export LD_LIBRARY_PATH="$PWD/build/src:$LD_LIBRARY_PATH"
    Note that only attention with causal mask or no mask is supported.
"""

@ir.transform.module_pass(opt_level=0)
class MHARewritePass(relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def transform_module(self, mod, ctx):
        mod = relay.transform.CanonicalizeOps()(mod)
        func = rewrite([MHASlicePattern(), MHASplitPattern()], mod["main"])
        mod.update_func(mod.get_global_var("main"), func)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.MergeComposite(pattern_table())(mod)
        mod = relay.transform.AnnotateTarget("MHA")(mod)
        mod = relay.transform.PartitionGraph(bind_constants=False)(mod)
        mod = relay.transform.InferType()(mod)
        return mod

class MHASlicePattern(DFPatternCallback):
    def __init__(self):
        super().__init__(True, False)
        self.pattern = self.make_attention_pattern()

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.container.Map) -> relay.Expr:
        stacked_qkv = node_map[self.stacked_qkv][0]
        out_shape = pre.checked_type.shape
        attention_scale = float(node_map[self.attention_scale][0].data.numpy())
        if node_map[self.scaled_attention_scores][0].op.name == "divide":
            attention_scale = 1 / attention_scale
        num_head = node_map[self.attention_probs][0].checked_type.shape[1]
        attrs = ir.make_node("DictAttrs",
                             batch=out_shape[0], seq_len=out_shape[1], hidden_dim=out_shape[2],
                             num_head=num_head, softmax_scale=attention_scale)
        return relay.Call(relay.op.get("Attention"), [stacked_qkv], attrs)

    def make_attention_pattern(self, with_bias:bool = False) -> relay.Pattern:
        """Create pattern for fused MHA."""
        self.stacked_qkv = wildcard()
        sliced_q = is_op("strided_slice")(self.stacked_qkv)
        sliced_k = is_op("strided_slice")(self.stacked_qkv)
        sliced_v = is_op("strided_slice")(self.stacked_qkv)
        q = is_op("transpose")(is_op("reshape")(sliced_q))
        k = is_op("transpose")(is_op("reshape")(sliced_k))
        v = is_op("transpose")(is_op("reshape")(sliced_v))
        self.attention_scores = is_op("welder.matmul")(q, k)
        self.attention_scale = is_constant()
        self.scaled_attention_scores = (is_op("multiply") | is_op("divide"))(self.attention_scores, self.attention_scale)
        if with_bias:
            self.bias = wildcard()
            attention_scores = is_op("add")(self.scaled_attention_scores, self.bias)
        else:
            attention_scores = self.scaled_attention_scores
        self.attention_probs = is_op("nn.softmax")(attention_scores)
        self.context_layer = is_op("welder.matmul")(self.attention_probs, v)
        self.context_layer_reshaped = is_op("reshape")(is_op("transpose")(self.context_layer))
        return self.context_layer_reshaped

class MHASplitPattern(DFPatternCallback):
    def __init__(self):
        super().__init__(True, False)
        self.pattern = self.make_attention_pattern()

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.container.Map) -> relay.Expr:
        stacked_qkv = node_map[self.stacked_qkv][0]
        out_shape = pre.checked_type.shape
        attention_scale = float(node_map[self.attention_scale][0].data.numpy())
        if node_map[self.scaled_attention_scores][0].op.name == "divide":
            attention_scale = 1 / attention_scale
        num_head = node_map[self.attention_probs][0].checked_type.shape[1]
        attrs = ir.make_node("DictAttrs",
                             batch=out_shape[0], seq_len=out_shape[1], hidden_dim=out_shape[2],
                             num_head=num_head, softmax_scale=attention_scale)
        return relay.Call(relay.op.get("Attention"), [stacked_qkv], attrs)

    def make_attention_pattern(self, with_bias:bool = False) -> relay.Pattern:
        """Create pattern for fused MHA."""
        self.stacked_qkv = wildcard()
        self.stacked_qkv_transformed = is_op("transpose")(is_op("reshape")(self.stacked_qkv))
        self.qkv_tuple = is_op("split")(self.stacked_qkv_transformed)
        sliced_q = is_tuple_get_item(self.qkv_tuple, 0)
        sliced_k = is_tuple_get_item(self.qkv_tuple, 1)
        sliced_v = is_tuple_get_item(self.qkv_tuple, 2)
        q = is_op("squeeze")(sliced_q)
        k = is_op("transpose")(is_op("squeeze")(sliced_k))
        v = is_op("squeeze")(sliced_v)
        self.attention_scores = is_op("welder.matmul")(q, k)
        self.attention_scale = is_constant()
        self.scaled_attention_scores = (is_op("multiply") | is_op("divide"))(self.attention_scores, self.attention_scale)
        if with_bias:
            self.bias = wildcard()
            attention_scores = is_op("add")(self.scaled_attention_scores, self.bias)
        else:
            attention_scores = self.scaled_attention_scores
        self.attention_probs = is_op("nn.softmax")(attention_scores)
        self.context_layer = is_op("welder.matmul")(self.attention_probs, v)
        self.context_layer_reshaped = is_op("reshape")(is_op("transpose")(self.context_layer))
        return self.context_layer_reshaped

@tvm._ffi.register_func("relay.ext.MHA")
def _compiler(func):
    attrs = func.body.op.body.attrs
    tvm_symbol = func.attrs["global_symbol"]
    code = codegen_MHA(tvm_symbol, attrs)
    csrc_module_create = tvm._ffi.get_global_func("runtime.CSourceModuleCreate")
    link_mod = csrc_module_create(code, "cc", [tvm_symbol], [])
    add_extern_lib("-lflash_attn")
    return link_mod

def op_relation_MHA(arg_types, attrs):
    out_shape = [attrs["batch"], attrs["seq_len"], attrs["hidden_dim"]]
    return relay.TensorType(out_shape, "float16")

def op_register_MHA():
    op_name = "Attention"
    reg.register(op_name)
    op = reg.get(op_name)
    op.set_num_inputs(1)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", op_relation_MHA)
    op.add_argument("x", "Tensor", "The Input data.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OPAQUE)

op_register_MHA()

@register_pattern_table("MHA")
def pattern_table() -> List[Tuple[str, relay.Pattern, Callable[[relay.Call], bool]]]:
    def attention_pattern() -> relay.Pattern:
        return is_op("Attention")(wildcard())

    def check_attention(matched: relay.Call) -> bool:
        return True

    return [
        ("MHA.stacked", attention_pattern(), check_attention),
    ]

def codegen_MHA(tvm_symbol: str, attrs: dict) -> str:
    temp_symbol = tvm_symbol + "_c"
    batch_size = attrs["batch"]
    seq_len = attrs["seq_len"]
    num_head = attrs["num_head"]
    hidden_dim = attrs["hidden_dim"]
    head_dim = hidden_dim // num_head
    batch_stride = 3 * hidden_dim * seq_len
    out_batch_stride = hidden_dim * seq_len
    softmax_scale = float(attrs["softmax_scale"])
    q_ptr = "(cutlass::half_t*)x->data"
    k_ptr = f"(cutlass::half_t*)x->data+{hidden_dim}"
    v_ptr = f"(cutlass::half_t*)x->data+{2*hidden_dim}"
    output_ptr = "(cutlass::half_t*)y->data"
    template = f"""
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
#include "flash.h"

extern "C" int {temp_symbol}(DLTensor* x, DLTensor* y) {{
    flash_attn::flash_attention_forward({q_ptr}, {k_ptr}, {v_ptr}, {output_ptr},
    {batch_size}, {seq_len}, {seq_len}, {num_head}, {num_head}, {head_dim},
    {batch_stride}, {batch_stride}, {batch_stride}, {out_batch_stride},
    {head_dim}, {head_dim}, {head_dim}, {head_dim},
    {3*hidden_dim}, {3*hidden_dim}, {3*hidden_dim}, {hidden_dim},
    {softmax_scale}, false, 0);
    return 0;
}}

TVM_DLL_EXPORT_TYPED_FUNC({tvm_symbol}, {temp_symbol});
"""
    return template
