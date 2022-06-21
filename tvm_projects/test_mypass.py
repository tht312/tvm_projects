# -*- coding: utf-8 -*-
import tvm
import tvm.relay as relay
from mypass4_3 import MyPass
import numpy as np
import gc

from pass_reset_input import mod, params

# print("before:",mod['main'])
seq = tvm.transform.Sequential(
    [
          MyPass()
    ]
)
mod = seq(mod)
# print("after:",mod["main"])

from pass_for_device import AddDeviceCopy
seq = tvm.transform.Sequential(
    [
          AddDeviceCopy(),
    ]
)
mod = seq(mod)
# print("final:",mod["main"])

target = {"cpu": "llvm", "cuda": "cuda"}
ctx_cpu = tvm.cpu(0)
ctx_gpu = tvm.cuda(0)
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, target_host="llvm", params=params)
m = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx_cpu,ctx_gpu))
data = np.random.uniform(0,1,size=(1,3,25000,25000)).astype(np.float32)
m.set_input("input0", tvm.nd.array(data))
del data
gc.collect()

import time
start = time.perf_counter()
m.run()
tvm_output = m.get_output(0)
print(tvm_output)
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))
