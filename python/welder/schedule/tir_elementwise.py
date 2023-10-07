import numpy as np
from tvm import tir
import tvm
import os
from ..config import Stride
from .tir_base import TIRSchedulerBase


class TIRElementWiseScheduler(TIRSchedulerBase):
    def schedule(self) -> tir.Schedule:
        
        sch, config = self.sche, self.config
        self.schedule_compute_inline()
        self.block_size[0] = int(np.prod(config.thread))

        C = sch.get_block(self.output_op.name)
        space_loops = sch.get_loops(C)[:len(self.output_op.axis)]        

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, loop in enumerate(space_loops):
            if self.output_op.axis[i].dom.extent % config.block[i]:
                raise NotImplementedError("Undivisible block in TIR schedule is still buggy.")
            bx, _t = sch.split(loop, factors=[None, config.block[i]])
            vx, _t = sch.split(
                _t, factors=[None, config.thread[i] * config.step[i]])
            tx, tn = sch.split(_t, factors=[None, config.step[i]])
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)
        vthd_axis = list(reversed(vthd_axis))  # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
        sch.reorder(*axis_order)
        blck_fused = sch.fuse(*blck_axis)
        thrd_fused = sch.fuse(*thrd_axis)
        sch.bind(blck_fused, "blockIdx.x")
        sch.bind(thrd_fused, "threadIdx.x")
        if len(vthd_axis) > 3:
            vthd_axis = vthd_axis[0:2] + [sch.fuse(*vthd_axis[2:])]
        for i, ax in enumerate(vthd_axis):
            sch.bind(ax, "vthread" + ['.x', '.y', '.z'][i])
        for ax in tile_axis:
            sch.unroll(ax)

        # ----- cache small tensors -----
        cached_stages = []
        for i, input_tensor in enumerate(self.output_op.input_tensors):
            cached_stages.append(input_tensor.name)

        cache_plan = self.make_cache_plan()

        for tensor in cache_plan:
            tensor_shared = sch.cache_read(C, tensor.name, "shared")
            sch.compute_at(tensor_shared, thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            dim_offset = len(vthd_axis) + 2 # outer loops are: blck_fused vthd_axis thrd_fused
            self.cooperative_fetch(tensor_shared, dim_offset, strides)
            if len(self.shared_outputs) == 0:
                continue
            tensor_local = sch.cache_read(C, tensor.name, "local")
            sch.compute_at(tensor_local, thrd_fused)

        return sch.mod["main"]
