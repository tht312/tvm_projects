# -*- coding: utf-8 -*-
import tvm
import tvm.relay as relay

from tvm.relay.prelude import Prelude
def run_opt_pass(expr, opt_pass, import_prelude=False):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    if import_prelude:
        Prelude(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def run_infer_type(expr):
    return run_opt_pass(expr, relay.transform.InferType())

@relay.transform.function_pass(opt_level=1)
class AddDeviceCopy:
    """Simple test function to replace one argument to another."""

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):

        class ReplaceConv2d(tvm.relay.ExprMutator):
            def visit_call(self, call):
                new_args = [self.visit(arg) for arg in call.args]
                new_fn = self.visit(call.op)
                
                if(call.op == tvm.ir.Op.get('nn.conv2d') or call.op == tvm.ir.Op.get('nn.conv2d_transpose') or call.op == tvm.ir.Op.get('nn.conv2d_bw_filter')): # or call.op == tvm.ir.Op.get('nn.conv2d_bw_filter')
                    data = relay.device_copy(new_args[0],"cpu","cuda")
                    weight = relay.device_copy(new_args[1],"cpu","cuda")
                    new_args = [data,weight]
                    output = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)
                    return relay.device_copy(output,"cuda","cpu")
                        
                return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
            
        return ReplaceConv2d().visit(func)
