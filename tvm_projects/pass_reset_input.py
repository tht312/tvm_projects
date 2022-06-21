# -*- coding: utf-8 -*-
import torch
import tvm
import tvm.relay as relay
import gc

@relay.transform.function_pass(opt_level=1)
class ResetInput:
    """Simple test function to replace one argument to another."""

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):

        class ReplaceVar(tvm.relay.ExprMutator):
            def visit_var(self, var):
                if (var.name_hint == "input0"):
                    return relay.var("input0", shape=[1,3,6800,7200])
                else:
                    return var
            
        return ReplaceVar().visit(func)
    

from fcn_for_train import FCN32s
model = FCN32s(n_class=6)
# checkpoint = torch.load('../torch/torch_model/fcn32s_99.pth')
checkpoint = torch.load('fcn32s_99.pth')
model.load_state_dict(checkpoint)
model.eval()

input_shape = [1, 3, 300, 500]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

input_name = "input0"
shape_list = [(input_name, torch.Size([1, 3, 300, 500]))]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

seq = tvm.transform.Sequential(
    [
         ResetInput()
    ]
)
mod = seq(mod)

del model,scripted_model
gc.collect()
torch.cuda.empty_cache()
