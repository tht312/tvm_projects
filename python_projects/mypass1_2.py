# -*- coding: utf-8 -*-
import tvm
import tvm.relay as relay
import numpy as np

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def run_infer_type(expr):
    return run_opt_pass(expr, relay.transform.InferType())

target = {"cpu": "llvm", "cuda": "cuda"}
ctx_cpu = tvm.cpu(0)
ctx_gpu = tvm.cuda(0)

@relay.transform.function_pass(opt_level=1)
class MyPass:
    """Simple test function to replace one argument to another."""

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):

        class ReplaceCall(tvm.relay.ExprMutator):
            def visit_call(self, call):

                if(call.op == tvm.ir.Op.get('nn.conv2d')):
                    #print(call.attrs.keys())
                    padding = call.attrs['padding']
                    strides = call.attrs['strides']
                    channels = call.attrs['channels']
                    kernel_size = call.attrs['kernel_size']
                    P = padding[0]
                    S = strides[0]
                    K = kernel_size[0]
                    
                    if(isinstance(call.args[0],relay.expr.Var)):
                        ty = call.args[0].type_annotation
                        data_shape = list(ty.shape)
                        data_channels = data_shape[1]
                        height = data_shape[2]
                        width = data_shape[3]
                        
                        if(height*width >= 5000*4999):
                            # N = height*width // (3000*3000)
                            # N = int(N) + 1
                            N = 2

                            names = locals()
                            for i in range(N):
                                if (i == 0):
                                    new_width = width//N
                                    new_width = (new_width-K+P)//S*S+K-P
                                    x = relay.strided_slice(call.args[0],begin=[0,0,0,0],end=[1,data_channels,height,new_width],strides=[1,1,1,1])
                                    # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                    # y = relay.device_copy(call.args[1],ctx_cpu,ctx_gpu)
                                    names['z'+str(i)] = relay.nn.conv2d(x, call.args[1], strides=strides, padding=(P,P,P,0), channels=channels, kernel_size=kernel_size)
                                    # names['z'+str(i)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                    
                                elif (i == N-1):
                                    new_width = width//N*(N-1)
                                    new_width = (new_width-K+P)//S*S+K-P
                                    x = relay.strided_slice(call.args[0],begin=[0,0,0,new_width-K+S],end=[1,data_channels,height,width],strides=[1,1,1,1])
                                    # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                    names['z'+str(i)] = relay.nn.conv2d(x, call.args[1], strides=strides, padding=(P,0,P,P), channels=channels, kernel_size=kernel_size)
                                    # names['z'+str(i)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                    
                                else:
                                    old_width = width//N*i
                                    old_width = (old_width-K+P)//S*S+K-P
                                    new_width = width//N*(i+1)
                                    new_width = (new_width-K+P)//S*S+K-P
                                    x = relay.strided_slice(call.args[0],begin=[0,0,0,old_width-K+S],end=[1,data_channels,height,new_width],strides=[1,1,1,1])
                                    # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                    names['z'+str(i)] = relay.nn.conv2d(x, call.args[1], strides=strides, padding=(P,0,P,0), channels=channels, kernel_size=kernel_size)
                                    # names['z'+str(i)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                    
                                    
                            List = []
                            for i in range(N):
                                List.append(names['z'+str(i)])
            
                            return relay.Tuple(List)
                        else:
                            return call
                            
                    else:
                        Tuple = self.visit(call.args[0])
                        if(isinstance(Tuple,relay.Tuple)):
                            N = len(Tuple)
                            count = 0
                            names = locals()
                            
                            for i in range(N):
                                names['x'+str(i)+'shape'] = run_infer_type(Tuple[i]).checked_type
                                
                            for x in Tuple.fields:
                                if (count == 0):
                                    C = names['x'+str(count)+'shape'].concrete_shape[1]
                                    H = names['x'+str(count)+'shape'].concrete_shape[2]
                                    W = names['x'+str(count)+'shape'].concrete_shape[3]
                                    W = (W-K+P)//S*S+K-P
                                    
                                    x = relay.strided_slice(Tuple[count],begin=[0,0,0,0],end=[1,C,H,W],strides=[1,1,1,1])
                                    # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                    # y = relay.device_copy(call.args[1],ctx_cpu,ctx_gpu)
                                    names['z'+str(count)] = relay.nn.conv2d(x,call.args[1],strides=(S,S), padding=(P,P,P,0), channels=channels, kernel_size=(K,K))
                                    # names['z'+str(count)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                    
                                    count = count + 1
                                        
                                elif(count == N-1):
                                    C = names['x'+str(count-1)+'shape'].concrete_shape[1]
                                    H = names['x'+str(count-1)+'shape'].concrete_shape[2]
                                    W = names['x'+str(count-1)+'shape'].concrete_shape[3]
                                    
                                    if(K != 1):
                                        x_left = relay.strided_slice(Tuple[count-1],begin=[0,0,0,W-K+S],end=[1,C,H,W],strides=[1,1,1,1])
                                        x = relay.concatenate((x_left,x),3)
                                        
                                    # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                    names['z'+str(count)] = relay.nn.conv2d(x,call.args[1],strides=(S,S), padding=(P,0,P,P), channels=channels, kernel_size=(K,K))
                                    # names['z'+str(count)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                    
                                else:
                                    C = names['x'+str(count)+'shape'].concrete_shape[1]
                                    H = names['x'+str(count)+'shape'].concrete_shape[2]
                                    W = names['x'+str(count)+'shape'].concrete_shape[3]
                                    W_left = names['x'+str(count-1)+'shape'].concrete_shape[3]
                                    
                                    if(K != 1):
                                        x_left = relay.strided_slice(Tuple[count-1],begin=[0,0,0,W_left-K+S],end=[1,C,H,W_left],strides=[1,1,1,1])
                                        x = relay.concatenate((x_left,x),3)
                                    
                                    W = W+K-S
                                    W = (W-K+P)//S*S+K-P
                                    x = relay.strided_slice(x,begin=[0,0,0,0],end=[1,C,H,W],strides=[1,1,1,1])
                                    # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                    names['z'+str(count)] = relay.nn.conv2d(x,call.args[1],strides=(S,S), padding=(P,0,P,0), channels=channels, kernel_size=(K,K))
                                    # names['z'+str(count)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                    
                                    count = count+1
                        
                            List = []
                            for i in range(N):
                                List.append(names['z'+str(i)])

                            return relay.Tuple(List)
                        
                        else:
                            return call
                        
                if(call.op == tvm.ir.Op.get('nn.bias_add')):
                    Tuple = self.visit(call.args[0])
                    if(isinstance(Tuple,relay.Tuple)):
                        N = len(Tuple)
                        count = 0
                        names = locals()
                        
                        for x in Tuple.fields:
                            names['z'+str(count)] = relay.nn.bias_add(x,call.args[1])
                            count = count+1
                            
                        List = []
                        for i in range(N):
                            List.append(names['z'+str(i)])
                        return relay.Tuple(List)
                    else:
                        return call
                
                if(call.op == tvm.ir.Op.get('nn.relu')):
                    Tuple = self.visit(call.args[0])
                    if(isinstance(Tuple,relay.Tuple)):
                        N = len(Tuple)
                        count = 0
                        names = locals()
                        
                        for x in Tuple.fields:
                            names['z'+str(count)] = relay.nn.relu(x)
                            count = count+1
                            
                        List = []
                        for i in range(N):
                            List.append(names['z'+str(i)])
                        return relay.Tuple(List)
                    else:
                        return call
                
                if(call.op == tvm.ir.Op.get('nn.max_pool2d')):
                    Tuple = self.visit(call.args[0])
                    pool_size = call.attrs['pool_size']
                    strides = call.attrs['strides']
                    padding = call.attrs['padding']
                    ceil_mode = call.attrs['ceil_mode']
                    
                    if(isinstance(Tuple,relay.Tuple)):
                        N = len(Tuple)
                        count = 0
                        names = locals()
                        
                        for i in range(N):
                            if (i == 0):
                                shape_0 = run_infer_type(Tuple[i]).checked_type
                                C = shape_0.concrete_shape[1]
                                H = shape_0.concrete_shape[2]
                                names['x'+str(i)+'W'] = shape_0.concrete_shape[3]
                            else:
                                names['x'+str(i)+'W'] = names['x'+str(i-1)+'W'] + run_infer_type(Tuple[i]).checked_type.concrete_shape[3]
                        
                        for x in Tuple.fields:
                            if (count == 0):
                                W = names['x'+str(count)+'W']
                                if(W % 2 == 0):
                                    names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                                    count = count+1
                                else:
                                    x = relay.strided_slice(x,begin=[0,0,0,0],end=[1,C,H,W-1],strides=[1,1,1,1])
                                    names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                                    count = count+1
                                    
                            elif (count == N-1):
                                W_left_all = names['x'+str(count-1)+'W']
                                W = names['x'+str(count)+'W'] - names['x'+str(count-1)+'W']
                                if (count == 1):
                                    W_left = names['x'+str(count-1)+'W']
                                else:
                                    W_left = names['x'+str(count-1)+'W'] - names['x'+str(count-2)+'W']
                                if(W_left_all % 2 == 0):
                                    names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                                else:
                                    x_left = relay.strided_slice(Tuple[count-1],begin=[0,0,0,W_left-1],end=[1,C,H,W_left],strides=[1,1,1,1])
                                    x = relay.concatenate((x_left,x),3)
                                    names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                            
                            else:
                                W_left_all = names['x'+str(count-1)+'W']
                                W_right_all = names['x'+str(count)+'W']
                                W = W_right_all - W_left_all
                                if (count == 1):
                                    W_left = names['x'+str(count-1)+'W']
                                else:
                                    W_left = names['x'+str(count-1)+'W'] - names['x'+str(count-2)+'W']
                                if (W_left_all % 2 == 0):
                                    if (W_right_all % 2 == 0):
                                        names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                                        count = count+1
                                    else:
                                        x = relay.strided_slice(x,begin=[0,0,0,0],end=[1,C,H,W-1],strides=[1,1,1,1])
                                        names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                                        count = count+1
                                else:
                                    x_left = relay.strided_slice(Tuple[count-1],begin=[0,0,0,W_left-1],end=[1,C,H,W_left],strides=[1,1,1,1])
                                    x = relay.concatenate((x_left,x),3)
                                    if (W_right_all % 2 == 0):
                                        names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                                        count = count+1
                                    else:
                                        x = relay.strided_slice(x,begin=[0,0,0,0],end=[1,C,H,W+1-1],strides=[1,1,1,1])
                                        names['z'+str(count)] = relay.nn.max_pool2d(x,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                                        count = count+1
                                    
                        List = []
                        for i in range(N):
                            List.append(names['z'+str(i)])
                        return relay.Tuple(List)
                    
                    else:
                        return call
                    
                if(call.op == tvm.ir.Op.get("nn.conv2d_transpose")):
                    channels = call.attrs['channels']
                    kernel_size = call.attrs['kernel_size']
                    padding = call.attrs['padding']
                    strides = call.attrs['strides']
                    K = kernel_size[0]
                    P = padding[0]
                    S = int(strides[0])
                    
                    Tuple = self.visit(call.args[0])
                    if(isinstance(Tuple,relay.Tuple)):
                        N = len(Tuple)
                        count = 0
                        names = locals()
                        
                        for i in range(N):
                            names['x'+str(i)+'shape'] = run_infer_type(Tuple[i]).checked_type
                            
                        for x in Tuple.fields:
                            if (count == 0):
                                C = names['x0shape'].concrete_shape[1]
                                H = names['x0shape'].concrete_shape[2]
                                W = names['x0shape'].concrete_shape[3]
                                
                                x = relay.nn.dilate(x,strides=(1,1,S,S),dilation_value=0.0)
                                H = S*(H-1)+1
                                W = S*(W-1)+1
                                
                                constant = relay.const(np.zeros((1,C,H,S-1)))
                                x = relay.concatenate((x,constant),3)
                                # y = relay.device_copy(call.args[1],ctx_cpu,ctx_gpu)
                                y = relay.reverse(call.args[1],-1)
                                y = relay.reverse(y,-2)
                                y = relay.transpose(y,(1,0,2,3))
                                
                                names['x0'] = x
                                # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                #y = relay.device_copy(y,ctx_cpu,ctx_gpu)
                                names['z0'] = relay.nn.conv2d(x,y,strides=(1,1),padding=(K-P-1,K-P-1,K-P-1,0),channels=channels,kernel_size=(K,K))
                                # names['z0'] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                count = count+1
                                
                            elif (count == N-1):
                                W = names['x'+str(count-1)+'shape'].concrete_shape[3]
                                W = S*(W-1)+1
                                if (count-1 != 0):
                                    W = W+K-1
                                
                                x_left = relay.strided_slice(names['x'+str(count-1)],begin=[0,0,0,W+S-K],end=[1,C,H,W+S-1],strides=[1,1,1,1])
                                x = relay.nn.dilate(x,strides=(1,1,S,S),dilation_value=0.0)
                                x = relay.concatenate((x_left,x),3)
                                
                                # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                names['z'+str(count)] = relay.nn.conv2d(x,y,strides=(1,1),padding=(K-P-1,0,K-P-1,K-P-1),channels=channels,kernel_size=(K,K))
                                # names['z'+str(count)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                
                            else:
                                W = names['x'+str(count-1)+'shape'].concrete_shape[3]
                                W = S*(W-1)+1
                                if (count-1 != 0):
                                    W = W+K-1
                                
                                x_left = relay.strided_slice(names['x'+str(count-1)],begin=[0,0,0,W+S-K],end=[1,C,H,W+S-1],strides=[1,1,1,1])
                                x = relay.nn.dilate(x,strides=(1,1,S,S),dilation_value=0.0)
                                x = relay.concatenate((x_left,x),3)
                                x = relay.concatenate((x,constant),3)

                                names['x'+str(count)] = x
                                # x = relay.device_copy(x,ctx_cpu,ctx_gpu)
                                names['z'+str(count)] = relay.nn.conv2d(x,y,strides=(1,1),padding=(K-P-1,0,K-P-1,0),channels=channels,kernel_size=(K,K))
                                # names['z'+str(count)] = relay.device_copy(z,ctx_gpu,ctx_cpu)
                                count = count+1
                                
                        List = []
                        for i in range(N):
                            List.append(names['z'+str(i)])
                        return relay.Tuple(List)
                    else:
                        return call
                                
                if(call.op == tvm.ir.Op.get("nn.dropout")):
                    Tuple = self.visit(call.args[0])
                    if(isinstance(Tuple,relay.Tuple)):
                        N = len(Tuple)
                        count = 0
                        names = locals()
                        
                        for x in Tuple.fields:
                            names['z'+str(count)] = relay.nn.dropout(x)
                            count = count+1
                            
                        List = []
                        for i in range(N):
                            List.append(names['z'+str(i)])
                        return relay.Tuple(List)
                    else:
                        return call
                
                return call
            
            def visit_function(self, fn):
                new_params = [self.visit(x) for x in fn.params]
                new_body = self.visit(fn.body)
                if(isinstance(new_body,relay.Tuple)):
                    return relay.Function(list(new_params), relay.concatenate(new_body,3))
                else:
                    return relay.Function(list(new_params), new_body)
                
            def visit_tuple_getitem(self, op):
                tuple_value = self.visit(op.tuple_value)
                if not tuple_value.same_as(op.tuple_value):
                    return tuple_value
                return op
            
        return ReplaceCall().visit(func)

# def example():
#     x = relay.var("x", shape=[1,1,3000,9002])
#     y = relay.var("y", shape=[1,1,3,3])
#     z = relay.nn.conv2d(x, y, strides=(1,1), padding=(1,1), channels=1, kernel_size=(3,3))
#     z = relay.nn.conv2d(z, y, strides=(1,1), padding=(1,1), channels=1, kernel_size=(3,3))
#     bias = relay.var("bias", shape=[1],dtype="float32")
#     bias_add = relay.nn.bias_add(z,bias)
#     relu = relay.nn.relu(bias_add)
#     maxpool = relay.nn.max_pool2d(relu,pool_size=(2,2),strides=(2,2),padding=(0,0,0,0),ceil_mode=True)
#     trans_conv = relay.nn.conv2d_transpose(maxpool,y,strides=(2,2),padding=(0,0,0,0),channels=1,kernel_size=(3,3))
#     dropout = relay.nn.dropout(trans_conv)
#     func = relay.Function(relay.analysis.free_vars(dropout), dropout)
#     return func

# f = example()
# print("before:", f)
# mod = tvm.IRModule.from_expr(f)
# seq = tvm.transform.Sequential(
#     [
#          #relay.transform.SimplifyInference(),
#          relay.transform.InferType(),
#          MyPass()
#     ]
# )
# mod1 = seq(mod)
# print("after:",mod1["main"])

# tensor = tvm.nd.array(np.random.uniform(0, 1, size=(1,1,3000,9002)).astype(np.float32))
# weight = tvm.nd.array(np.random.uniform(0, 1, size=(1,1,3,3)).astype(np.float32))
# bias = tvm.nd.array(np.random.uniform(0, 1, size=(1)).astype(np.float32))
# with tvm.transform.PassContext(opt_level=3):
#    lib = relay.build(mod1, target, target_host="llvm")
# module = tvm.contrib.graph_runtime.GraphModule(lib["default"](ctx_cpu,ctx_gpu))
# module.set_input("x", tensor)
# module.set_input("y", weight)
# module.set_input("bias", bias)
# module.run()
# out = module.get_output(0).asnumpy()

# result = tvm.relay.create_executor(kind="vm",mod=mod).evaluate()(tensor,weight,bias).asnumpy()
# print(np.all(abs(result-out) < 0.01))
# print("optimized",out)
# print("original",result)
