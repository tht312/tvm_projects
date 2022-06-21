import tvm
from tvm import relay
import numpy as np

from gradient_ import tensor_to_tvm

class Module:
    def __init__(self, params, fw_mod, fw_and_bw_mod, drop_info, out_type):
        # self.params = {n: tensor_to_tvm(t) for n, t in params.items()}
        self.params = params
        self.optim = None

        self.target = {"cpu": "llvm", "cuda": "cuda"}
        self.target_host = "llvm"
        self.ctx_cpu = tvm.cpu(0)
        self.ctx_gpu = tvm.cuda(0)
        
        # self.fw_module = self.compile(fw_mod)
        self.fw_and_bw_module = self.compile(fw_and_bw_mod)
        
        self.drop_info = drop_info
        self.out_type = out_type
    
    def compile(self, mod):
        with tvm.transform.PassContext(opt_level=0,disabled_pass=["SimplifyInference","OpFusion"]): # disabled_pass=["SimplifyInference","OpFusion"]
            lib = relay.build(mod, self.target, self.target_host, params={})
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](self.ctx_cpu, self.ctx_gpu))
        
        return module

    def set_optim(self, optimizer):
        self.optim = optimizer
    
    def gen_dropout(self, drop_info):
        import torch
        
        drop_c = {}
        for k in drop_info.keys(): # we don't know the order
            # print(drop_info[k])
            p, typ = drop_info[k]
            drop_c[k] = torch.nn.functional.dropout(torch.ones([int(i) for i in typ.shape], 
                                                      dtype=getattr(torch, typ.dtype), device="cuda"), p=p)*(1-p)
        
        drop_tvm = {n: tensor_to_tvm(t) for n, t in drop_c.items()}
        # print(drop_tvm)
        
        return drop_tvm
    
    def gen_grad_out(self, output_type):
        if(not isinstance(output_type, tvm.relay.TensorType)):
            # grad_outs = tuple(np.random.uniform(0, 1, size=t.shape).astype(np.float32) for i, t in enumerate(output_type.fields))
            grad_outs = tuple(np.ones(shape=t.shape, dtype=np.float32) for i, t in enumerate(output_type.fields))
        else:
            new_shape=[]
            for i in output_type.shape:
                new_shape.append(int(i))
            grad_outs = np.ones(shape=new_shape, dtype=np.float32)
        return grad_outs
    
    def train(self, **inputs):
        self.fw_and_bw_module.set_input(**self.params)
        
        drop_tvm = self.gen_dropout(self.drop_info)
        self.fw_and_bw_module.set_input(**drop_tvm)
        
        grad_outs = self.gen_grad_out(self.out_type)
        if isinstance(self.out_type, tvm.relay.TensorType):
            grad_outs_tvm = {"gr:out:0": tvm.nd.array(grad_outs)}
        else:
            grad_outs_tvm = {f"gr:out:{i}": tvm.nd.array(go) for i, go in enumerate(grad_outs)}
        self.fw_and_bw_module.set_input(**grad_outs_tvm)

        # inputs_tvm = [tensor_to_tvm(t) for t in inputs]
        # for n, i in zip(self.fw_inp_name, inputs):
        #     self.fw_and_bw_module.set_input(n, i)
        self.fw_and_bw_module.set_input(**inputs)
        
        self.fw_and_bw_module.run()
        
        if isinstance(self.out_type, tvm.relay.TensorType):
            res = self.fw_and_bw_module.get_output(0)
            # num_outputs = 1
        else:
            res = tuple(self.fw_and_bw_module.get_output(i)
                        for i in range(len(self.out_type.fields)))
            # num_outputs = len(res)
        
        grad_in = [self.fw_and_bw_module.get_output(i) for i in range(2, self.fw_and_bw_module.get_num_outputs()-1)]
        # print([g.shape for g in grad_in])
        # print([p for p in self.params])
        if(self.optim is not None):
            new_param = self.optim.step(self.params, grad_in)
            self.params = {n: tvm.nd.array(p) for n, p in zip(self.params, new_param)}
        else:
            print("u have not defined an optimizer!")
        
        return res, grad_in
    
    def infer(self, **inputs):
        self.fw_module.set_input(**self.params)
        
        # drop_tvm = self.gen_dropout(self.drop_info)
        # self.fw_module.set_input(**drop_tvm)
        
        # inputs_tvm = [tensor_to_tvm(t) for t in inputs]
        # for n, i in zip(self.fw_inp_name, inputs):
        #     self.fw_and_bw_module.set_input(n, i)
        self.fw_module.set_input(**inputs)
        
        self.fw_module.run()
        
        if isinstance(self.out_type, tvm.relay.TensorType):
            res = self.fw_module.get_output(0)
        else:
            res = tuple(self.fw_module.get_output(i)
                        for i in range(len(self.out_type.fields)))
        
        return res
    
    def get_params(self):
         return self.params