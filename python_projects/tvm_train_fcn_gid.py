# -*- coding: utf-8 -*-
import tvm
import tvm.relay as relay
from mypass4_3 import MyPass
import numpy as np
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

import time
start = time.time()

label_width = 7200
label_height = 6800
n_class = 6

from pass_reset_input import mod, params
# from load_params import params
param_bytes = bytearray(open("GID/fcn_epoch_10", "rb").read())
load_params = tvm.runtime.load_param_dict(param_bytes)
new_params = {}
for p in params:
    new_params[p] = load_params[p]

seq = tvm.transform.Sequential(
    [
          MyPass(),
    ]
)
mod = seq(mod)

f = mod["main"].body
f = relay.strided_slice(f,begin=[0,0,19,19],end=[1,n_class,label_height+19,label_width+19],strides=[1,1,1,1])
transpose = relay.transpose(f,(0,2,3,1))
reshape = relay.reshape(transpose,(-1,n_class))
softmax = relay.nn.softmax(reshape,1)
eps = relay.const(1e-8)
softmax = relay.add(softmax,eps)
label = relay.var('label',shape=[label_width*label_height,n_class])
loss = relay.nn.cross_entropy(softmax,label)
mod = tvm.IRModule.from_expr(loss)
# print(relay.transform.InferType()(mod),flush=True)
end = time.time()
print("生成正向图时间:",end - start,flush=True)

from gradient_ import create_function
from module_ import Module
from optimizer import Adam
info = create_function(mod)

from pass_for_device import AddDeviceCopy
fw_and_bw_mod = info['fw_and_bw_mod']
seq = tvm.transform.Sequential(
    [
        AddDeviceCopy(),
    ]
)
fw_and_bw_mod = seq(fw_and_bw_mod)
info['fw_and_bw_mod'] = fw_and_bw_mod
# print('fw_and_bw_mod:',fw_and_bw_mod,flush=True)
import datetime
print(datetime.datetime.now(),"mypass finished",flush=True)

end1 = time.time()
print('生成反向图时间: ',end1 - end,flush=True)

module = Module(new_params, **info)
module.set_optim(Adam(lr=0.00001))

end2 = time.time()
print('编译时间: ',end2 - end1,flush=True)

import gc
import gdal
from gdalconst import GA_ReadOnly
import random

images = os.listdir("C:/Users/Administrator/spyder_projects/GID/image_RGB")
labels = os.listdir("C:/Users/Administrator/spyder_projects/GID/new_label")
image_dir = "C:/Users/Administrator/spyder_projects/GID/image_RGB/"
label_dir = "C:/Users/Administrator/spyder_projects/GID/new_label/"
images = images[:120]
random.shuffle(images)

loss_history = 100
total_loss = 0
for i in range(120):
    dataset = gdal.Open(image_dir+images[i],GA_ReadOnly)
    img_array = dataset.ReadAsArray()
    img_array = np.expand_dims(img_array, 0).astype(np.float32)
    dataset = gdal.Open(label_dir+images[i][:-4]+"_label.tif",GA_ReadOnly)
    label_array = dataset.ReadAsArray()
    label_array = np.expand_dims(label_array, 0)
    label_array = np.reshape(label_array,(label_width*label_height)).astype(int)
    label_array = np.eye(n_class)[label_array].astype(np.float32)
    inp = {"input0": tvm.nd.array(img_array),"label": tvm.nd.array(label_array)}
    del img_array,label_array
    gc.collect()
    
    res, grad = module.train(**inp)
    print(res,flush=True)
    end3 = time.time()
    print("i=",i,flush=True)
    print('执行时间: ',end3 - end2,flush=True)
    
    param_bytes = tvm.runtime.save_param_dict(module.get_params())
    with open(os.path.join("GID/fcn.param_{}".format(i)), "wb") as fo:
        fo.write(param_bytes)

    if(res.asnumpy() < loss_history):
        new_params = module.get_params()
    loss_history = res.asnumpy()
    
    total_loss = total_loss + res.asnumpy()
    
print("total loss:",total_loss)
