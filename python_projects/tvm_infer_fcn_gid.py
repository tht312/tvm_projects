# -*- coding: utf-8 -*-
import tvm
import tvm.relay as relay
import numpy as np
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

param_bytes = bytearray(open("../model/GID/fcn_epoch_130", "rb").read())
load_params = tvm.runtime.load_param_dict(param_bytes)

from mypass1_2 import MyPass
from pass_reset_input import mod, params
# print(params.keys())
new_params = {}
for p in params:
    new_params[p] = load_params[p]
# print(new_params.keys())

# print("before:",mod['main'])
seq = tvm.transform.Sequential(
    [
          MyPass()
    ]
)
mod = seq(mod)
# print("after:",mod["main"])

f = mod["main"].body
f = relay.strided_slice(f,begin=[0,0,19,19],end=[1,6,6800+19,7200+19],strides=[1,1,1,1])
transpose = relay.transpose(f,(0,2,3,1))
reshape = relay.reshape(transpose,(-1,6))
softmax = relay.nn.softmax(reshape,1)
mod = tvm.IRModule.from_expr(softmax)

from pass_for_device import AddDeviceCopy
seq = tvm.transform.Sequential(
    [
          AddDeviceCopy(),
          relay.transform.SimplifyInference()
    ]
)
mod = seq(mod)
# print("final:",mod["main"])

target = {"cpu": "llvm", "cuda": "cuda"}
ctx_cpu = tvm.cpu(0)
ctx_gpu = tvm.cuda(0)
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, target_host="llvm", params=new_params)
m = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx_cpu,ctx_gpu))

def writeTif(bands, path):
    if bands is None or bands.__len__() == 0:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
                
images = os.listdir("../dataset/image_RGB")
images = images[120:]

import gdal
from gdalconst import GA_ReadOnly
for i in range(30):
    dataset = gdal.Open("../dataset/image_RGB/"+images[i],GA_ReadOnly)
    img_array = dataset.ReadAsArray()
    img_array = np.expand_dims(img_array, 0)
    data = img_array.astype(np.float32)

    m.set_input("input0", tvm.nd.array(data))
    m.run()
    tvm_output = m.get_output(0).asnumpy()
    print(tvm_output.shape)
    img = np.argmax(tvm_output,axis=1)
    print(img.shape)

    img = np.reshape(img,(6800,7200))
    colormap = [[0,0,0],[255,0,0],[0,255,0],[0,255,255],[255,255,0],[0,0,255]]
    img = np.array(colormap)[img].astype(np.uint8)

    bands_data = []
    bands_data.append(img[:,:,0])
    bands_data.append(img[:,:,1])
    bands_data.append(img[:,:,2])
                
    writeTif(bands_data, "../test_images_1/gid_test_{}.tif".format(i))
