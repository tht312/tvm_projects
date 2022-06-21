# -*- coding: utf-8 -*-
import tvm
import tvm.relay as relay
from fcn_for_train import FCN32s
# import torch
from mypass4_3 import MyPass
import numpy as np
import gc
# import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_name = "fcn32s"
model = FCN32s(n_class=6)
#checkpoint = torch.load('model_best.pth.tar')
#model.load_state_dict(checkpoint['model_state_dict'])
#model.eval()

label_width = 7200
label_height = 6800
n_class = 6

from pass_reset_input import mod, params

# input_shape = [1, 3, label_height, label_width]
# input_data = torch.randn(input_shape)
# scripted_model = torch.jit.trace(model, input_data).eval()

# input_name = "input0"
# shape_list = [(input_name, torch.Size([1, 3, label_height, label_width]))]
# mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

f = mod["main"]
mod = tvm.IRModule.from_expr(f)
seq = tvm.transform.Sequential(
    [
          MyPass(),
    ]
)
mod = seq(mod)
# print(mod)

f = mod["main"].body
f = relay.strided_slice(f,begin=[0,0,19,19],end=[1,n_class,label_height+19,label_width+19],strides=[1,1,1,1])
transpose = relay.transpose(f,(0,2,3,1))
reshape = relay.reshape(transpose,(-1,n_class))
softmax = relay.nn.softmax(reshape,1)
label = relay.var('label',shape=[label_width*label_height,n_class])
loss = relay.nn.cross_entropy(softmax,label)
mod = tvm.IRModule.from_expr(loss)
# print(relay.transform.InferType()(mod))

from gradient_ import create_function
from module_ import Module
from optimizer import Adam

# ret_dict = dict()
# def _create_function(mod,ret):
#     ret['info'] = create_function(mod)
# build_thread = threading.Thread(target=_create_function,args=(mod,ret_dict))
# stack_size = threading.stack_size(1024*1024*64)
# build_thread.start()
# threading.stack_size(stack_size)
# build_thread.join()
# info = ret_dict['info']
info = create_function(mod)

# print(info['fw_mod'])
# print(info['fw_and_bw_mod'])

from pass_for_device import AddDeviceCopy
fw_and_bw_mod = info['fw_and_bw_mod']
seq = tvm.transform.Sequential(
    [
         AddDeviceCopy(),
    ]
)
fw_and_bw_mod = seq(fw_and_bw_mod)
info['fw_and_bw_mod'] = fw_and_bw_mod
# print('fw_and_bw_mod:',fw_and_bw_mod)

# device_dict = relay.analysis.collect_device_info(fw_and_bw_mod['main'])
# class deviceMutator(relay.ExprMutator):
#     def visit_call(self, call):
#         if(call in device_dict):
#             op_device = device_dict[call]
#             args_device = [device_dict[arg] for arg in call.args if arg in device_dict]
#             for i, ad in enumerate(args_device):
#                 if ad != op_device and call.op.name != "device_copy":
#                     print(call.op.name, "args", i)
#                     print("op device:", op_device)
#                     print({f"args{i}": d for i, d in enumerate(args_device)})
#                     print({f"args{i}": d.op.name for i, d in enumerate(call.args) if isinstance(d, tvm.relay.Call)})
#         return super().visit_call(call)
    
#     def visit_var(self, var):
#         return super().visit_var(var)
# temp = deviceMutator().visit(fw_and_bw_mod['main'])

module = Module(params, **info)
module.set_optim(Adam())

data = np.random.uniform(0,1,size=(1,3,label_height,label_width)).astype(np.float32)
label_data = np.random.randint(0,n_class,size=(1,label_height,label_width))
label_data = np.reshape(label_data,(label_width*label_height)).astype(int)
label_data = np.eye(n_class)[label_data].astype(np.float32)
inp = {'input0': tvm.nd.array(data),'label':tvm.nd.array(label_data)}

del data, label_data
gc.collect()

# res = module.infer(**inp)
# print(res)

res, grad = module.train(**inp)
print(res, grad)
