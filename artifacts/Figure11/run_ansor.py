import argparse
import os.path as osp
import numpy as np
import onnx
import tvm
import tvm.relay.testing
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor

def run_ansor(prefix, device, skip_tuning):
    target = tvm.target.cuda(arch="sm_70")
    onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    mod, params = relay.frontend.from_onnx(onnx_model)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=True)
    kern_count = 0
    total_size = 0
    for task, weight in zip(tasks, task_weights):
        kern_count += weight
        total_size += weight * int(np.prod(eval(task.workload_key)[-1])) * 4 # 4B float

    print("IRS: ", total_size / (1024 * 1024), "MB")
    print("Kernel Count: ", kern_count)
    log_file = osp.join(prefix, "ansor_tune.log")
    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)

    # Create graph executor
    dev = tvm.device(str(target), device)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.benchmark(dev, min_repeat_ms=500, end_to_end=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--skip', action="store_true")
    args = parser.parse_args()
    run_ansor(args.prefix, args.device, args.skip)
