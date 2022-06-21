from frontend import visualize
import tvm
import tvm.relay as relay
from mypass1_2 import MyPass
import numpy as np
import gc
import os
print(os.getpid())

def example():
    x = relay.var("x", shape=[1,3,5000,4999])
    y = relay.var("y", shape=[3,3,3,3])
    bias = relay.var("bias", shape=[3],dtype="float32")
    conv = relay.nn.conv2d(x, y, strides=(1,1), padding=(1,1), channels=3, kernel_size=(3,3))
    bias_add = relay.nn.bias_add(conv,bias)
    relu = relay.nn.relu(bias_add)
    y_1 = relay.var("y_1", shape=[3,3,3,3])
    bias_1 = relay.var("bias_1", shape=[3],dtype="float32")
    conv_1 = relay.nn.conv2d(relu, y_1, strides=(1,1), padding=(1,1), channels=3, kernel_size=(3,3))
    bias_add_1 = relay.nn.bias_add(conv_1,bias_1)
    relu_1 = relay.nn.relu(bias_add_1)
    maxpool = relay.nn.max_pool2d(relu_1,pool_size=(2,2),strides=(2,2),padding=(0,0,0,0),ceil_mode=True)
    # y_2 = relay.var("y_2", shape=[3,3,3,3])
    # trans_conv = relay.nn.conv2d_transpose(maxpool,y_2,strides=(2,2),padding=(0,0,0,0),channels=3,kernel_size=(3,3))
    # dropout = relay.nn.dropout(trans_conv)
    func = relay.Function(relay.analysis.free_vars(maxpool), maxpool)
    return func

f = example()
mod = tvm.IRModule.from_expr(f)

# from pass_reset_input import mod, params

# mod = relay.transform.InferType()(mod)
# print("before:", mod['main'])
dot = visualize(mod['main'])
dot.render("../output/before",format='png',view=True)

seq = tvm.transform.Sequential(
    [
          relay.transform.InferType(),
          MyPass()
    ]
)
mod = seq(mod)
# print("after:",mod["main"])
dot = visualize(mod['main'])
dot.render("../output/after",format='png',view=True)

from pass_for_device import AddDeviceCopy
seq = tvm.transform.Sequential(
    [
          AddDeviceCopy(),
          relay.transform.SimplifyInference()
    ]
)
mod = seq(mod)

target = {"cpu": "llvm", "cuda": "cuda"}
ctx_cpu = tvm.cpu(0)
ctx_gpu = tvm.cuda(0)
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, target_host="llvm", params=params)
m = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx_cpu,ctx_gpu))
data = np.random.uniform(0,1,size=(1,3,6800,7200)).astype(np.float32)
weight = np.random.uniform(0,1,size=(3,3,3,3)).astype(np.float32)
bias = np.random.uniform(0,1,size=(3)).astype(np.float32)
# m.set_input("x", tvm.nd.array(data))
# m.set_input("y", tvm.nd.array(weight))
# m.set_input("y_1", tvm.nd.array(weight))
# m.set_input("y_2", tvm.nd.array(weight))
# m.set_input("bias", tvm.nd.array(bias))
# m.set_input("bias_1", tvm.nd.array(bias))
# del data, weight, bias
# gc.collect()

m.set_input("input0", tvm.nd.array(data))

import time
start = time.perf_counter()
m.run()
tvm_output = m.get_output(0)
print(tvm_output.asnumpy().shape)
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))
