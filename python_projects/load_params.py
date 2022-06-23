# -*- coding: utf-8 -*-
import torch
from fcn import FCN32s
import tvm
import tvm.relay as relay
import gc
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_name = "fcn32s"
model = FCN32s(n_class=21)
checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

input_shape = [1, 3, 500, 500]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

input_name = "input0"
shape_list = [(input_name, torch.Size([1, 3, 500, 500]))]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

score_fr_weight = params['score_fr.weight'].asnumpy()[0:6,0:512,:,:]
fc7_weight = params['fc7.weight'].asnumpy()[0:512,0:512,:,:]
fc6_weight = params['fc6.weight'].asnumpy()[0:512,:,:,:]
upscore_weight = params['upscore.weight'].asnumpy()[0:6,0:6,:,:]

params['score_fr.weight'] = tvm.nd.array(score_fr_weight)
params['fc7.weight'] = tvm.nd.array(fc7_weight)
params['fc6.weight'] = tvm.nd.array(fc6_weight)
params['upscore.weight'] = tvm.nd.array(upscore_weight)

score_fr_bias = params['score_fr.bias'].asnumpy()[0:6]
fc7_bias = params['fc7.bias'].asnumpy()[0:512]
fc6_bias = params['fc6.bias'].asnumpy()[0:512]

params['score_fr.bias'] = tvm.nd.array(score_fr_bias)
params['fc7.bias'] = tvm.nd.array(fc7_bias)
params['fc6.bias'] = tvm.nd.array(fc6_bias)

# print(params['score_fr.weight'].asnumpy().shape)
# print(params['fc7.weight'].asnumpy().shape)
# print(params['fc6.weight'].asnumpy().shape)
# print(params['upscore.weight'].asnumpy().shape)

# print(params.keys())
# del(params['conv1_1.bias'],params['conv1_2.bias'],params['conv2_1.bias'],params['conv2_2.bias'],params['conv3_1.bias']
#     ,params['conv3_2.bias'],params['conv3_3.bias'],params['conv4_1.bias'],params['conv4_2.bias'],params['conv4_3.bias']
#     ,params['conv5_1.bias'],params['conv5_2.bias'],params['conv5_3.bias'],params['fc6.bias'],params['fc7.bias'],params['score_fr.bias'])
# print(params.keys())

del model,scripted_model,checkpoint
gc.collect()
torch.cuda.empty_cache()