from typing import Dict, List

from tvm import te

from ..config import Config, Stride
from ..te_utils import get_compute_ops, seperate_reduce_ops
from .tir_elementwise import *
from .te_elementwise import *
from .te_reduce import *
from .te_reduce_interthread import *
from .te_wmma import *
from .tir_mma import TIRCutlassMMAScheduler
from .tir_simt import *


def schedule(args: List[te.Tensor], config: Config, shared_inputs: List[te.Tensor] = [],
        shared_inputs_strides: Dict[te.Tensor, Stride] = {}, shared_outputs = []):
    ops = get_compute_ops(args)
    reduces_ops, _ = seperate_reduce_ops(ops)
    input_args, output_args = [], []
    for arg in args:
        if isinstance(arg.op, te.PlaceholderOp):
            input_args.append(arg)
        else:
            output_args.append(arg)
    schedule_on_inner_stage = False
    for tensor in args:
        if isinstance(tensor.op, te.ComputeOp) and tensor.name not in config.schedule_stages:
            schedule_on_inner_stage = True

    if len(reduces_ops) == 0:
        assert(not schedule_on_inner_stage)
        if len(output_args) > 1:
            template = TEElementWiseScheduler
        else:
            template = TIRElementWiseScheduler
    elif config.use_tc and config.use_cutlass:
        template = TIRCutlassMMAScheduler
    elif config.use_tc and not config.use_cutlass:
        if schedule_on_inner_stage: raise NotImplementedError("Schedule not implemented")
        template = TEWarpMMAScheduler
    elif any([t > 1 for t in config.reduce_thread]):
        if schedule_on_inner_stage: raise NotImplementedError("Schedule not implemented")
        template = TEReduceInterThreadScheduler
    else:
        if schedule_on_inner_stage:
            raise NotImplementedError("Schedule not implemented")
            template = TIRSIMTScheduler
        else:
            template = TEReduceScheduler

    scheduler = template(args, config)

    scheduler.shared_inputs = shared_inputs
    scheduler.shared_outputs = shared_outputs
    scheduler.shared_inputs_strides = {tensor: Stride() for tensor in shared_inputs}
    scheduler.shared_inputs_strides.update(shared_inputs_strides)

    scheduler.make_passes()
    scheduler.schedule()

    return scheduler
