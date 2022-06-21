# -*- coding: utf-8 -*-
import tvm
import tvm.relay as relay
import numpy as np

# def example():
#     x = relay.var("x", shape=[1,3,600,600])
#     y = relay.var("y", shape=[3,3,3,3])
#     bias = relay.var("bias", shape=[3],dtype="float32")
#     z = relay.nn.conv2d(x, y, strides=(1,1), padding=(1,1), channels=3, kernel_size=(3,3))
#     bias_add = relay.nn.bias_add(z,bias)
#     relu = relay.nn.relu(bias_add)
#     maxpool = relay.nn.max_pool2d(relu,pool_size=(2,2),strides=(2,2),padding=(0,0,0,0),ceil_mode=True)
#     trans_conv = relay.nn.conv2d_transpose(maxpool,y,strides=(2,2),padding=(0,0,0,0),channels=3,kernel_size=(3,3))
#     # dropout = relay.nn.dropout(trans_conv)
#     z = relay.nn.conv2d(trans_conv, y, strides=(1,1), padding=(1,1), channels=3, kernel_size=(3,3))
#     func = relay.Function(relay.analysis.free_vars(z), z)
#     return func

from tvm.relay.prelude import Prelude
def run_opt_pass(expr, opt_pass, import_prelude=False):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    if import_prelude:
        Prelude(mod)
    #mod = relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def run_infer_type(expr):
    return run_opt_pass(expr, relay.transform.InferType())

target = {"cpu": "llvm", "cuda": "cuda"}
# ctx_cpu = tvm.context("cpu")
# ctx_gpu = tvm.context("cuda")
ctx_cpu = tvm.cpu(0)
ctx_gpu = tvm.cuda(0)

@relay.transform.function_pass(opt_level=1)
class MyPass:
    """Simple test function to replace one argument to another."""

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):

        class ReplaceCall(tvm.relay.ExprMutator):
            def __init__(self):
                self.n = 5
                super().__init__()
                
            def visit_call(self, call):
                n = self.n

                if(call.op == tvm.ir.Op.get('nn.conv2d')):
                    #print(call.attrs.keys())
                    padding = call.attrs['padding']
                    strides = call.attrs['strides']
                    channels = call.attrs['channels']
                    kernel_size = call.attrs['kernel_size']
                    P0 = padding[0] #top
                    P1 = padding[1] #left
                    P2 = padding[2] #bottom
                    P3 = padding[3] #right
                    S0 = strides[0]
                    S1 = strides[1]
                    K0 = kernel_size[0]
                    K1 = kernel_size[1]
                    if(isinstance(call.args[0],relay.expr.Var)):
                        ty = call.args[0].type_annotation
                        data_shape = list(ty.shape)
                        data_channels = data_shape[1]
                        height = data_shape[2]
                        width = data_shape[3]

                        list_width = []
                        for i in range(n+1):
                            new_width = width//n*i
                            if (i==n):
                                new_width = width
                            list_width.append((new_width-K1+P1)//S1*S1+K1-P1)

                        list_height = []
                        for i in range(n+1):
                            new_height = height//n*i
                            if (i==n):
                                new_height = height
                            list_height.append((new_height-K0+P0)//S0*S0+K0-P0)

                        list_relay = []
                        for i in range(n):
                            for j in range(n):
                                h1 = list_height[i]
                                if (i!=0):
                                    h1 = h1-K0+S0
                                h2 = list_height[i+1]

                                w1 = list_width[j]
                                if (j!=0):
                                    w1 = w1-K1+S1
                                w2 = list_width[j+1]

                                list_relay.append(relay.strided_slice(call.args[0],begin=[0,0,h1,w1],end=[1,data_channels,h2,w2],strides=[1,1,1,1]))

                        #x1 = relay.strided_slice(call.args[0],begin=[0,0,0,0],end=[1,data_channels,height,new_width],strides=[1,1,1,1])
                        #x2 = relay.strided_slice(call.args[0],begin=[0,0,0,new_width-K+S],end=[1,data_channels,height,width],strides=[1,1,1,1])

                        # for i in range(n*n):
                        #     list_relay[i] = relay.device_copy(list_relay[i], ctx_cpu, ctx_gpu)

                        #x1 = relay.device_copy(x1,ctx_cpu,ctx_gpu)
                        #x2 = relay.device_copy(x2,ctx_cpu,ctx_gpu)

                        # y = relay.device_copy(call.args[1],ctx_cpu,ctx_gpu)

                        for i in range(n):
                            for j in range(n):
                                P0 = padding[0]  # top
                                P1 = padding[1]  # left
                                P2 = padding[2]  # bottom
                                P3 = padding[3]  # right
                                if (i!=0):
                                    P0 = 0
                                if (i!=n-1):
                                    P2 = 0
                                if (j!=0):
                                    P1 = 0
                                if (j!=n-1):
                                    P3 = 0
                                list_relay[i*n+j] = relay.nn.conv2d(list_relay[i*n+j], call.args[1], strides=strides, padding=(P0,P1,P2,P3), channels=channels, kernel_size=kernel_size)

                        #z1 = relay.nn.conv2d(x1, y, strides=strides, padding=(P,P,P,0), channels=channels, kernel_size=kernel_size)
                        #z2 = relay.nn.conv2d(x2, y, strides=strides, padding=(P,0,P,P), channels=channels, kernel_size=kernel_size)

                        # for i in range(n * n):
                        #     list_relay[i] = relay.device_copy(list_relay[i], ctx_gpu, ctx_cpu)

                        #z1 = relay.device_copy(z1,ctx_gpu,ctx_cpu)
                        #z2 = relay.device_copy(z2,ctx_gpu,ctx_cpu)
                        
                        return relay.Tuple(list_relay)
                    else:
                        Tuple = self.visit(call.args[0])
                        x1_shape = run_infer_type(Tuple[0]).checked_type
                        C = x1_shape.concrete_shape[1]
                        #H = x1_shape.concrete_shape[2]
                        #W1 = x1_shape.concrete_shape[3]
                        #W11 = (W1-K+P)//S*S+K-P

                        list_width = []
                        list_width1 = []
                        list_height = []
                        list_height1 = []
                        sum_w = 0
                        sum_h = 0

                        for i in range(n):
                            for j in range(n):
                                shape = run_infer_type(Tuple[i * n + j]).checked_type
                                W = shape.concrete_shape[3]
                                list_width.append(W)
                                if (j == 0):
                                    sum_w = 0
                                W1 = (sum_w+W-K1+P1)//S1*S1+K1-P1-sum_w
                                if (j == n - 1):
                                    W1 = W
                                list_width1.append(W1)
                                sum_w += W

                                H = shape.concrete_shape[2]
                                list_height.append(H)
                                H1 = (sum_h+H-K0+P0)//S0*S0+K0-P0-sum_h
                                if (i == n - 1):
                                    H1 = H
                                list_height1.append(H1)
                                if (j == n - 1):
                                    sum_h += H

                        dict_slice = {}
                        dict_zero = {}

                        list_relay = []
                        for i in range(n):
                            for j in range(n):
                                H1 = list_height1[i*n+j]
                                W1 = list_width1[i*n+j]
                                H = list_height[i*n+j]
                                W = list_width[i*n+j]
                                x1 = relay.strided_slice(Tuple[i*n+j],begin=[0,0,0,0],end=[1,C,H1,W1],strides=[1,1,1,1])
                                tuple12 = ((i,j),(i,j+1))
                                x12 = relay.strided_slice(Tuple[i*n+j],begin=[0,0,0,W1-K1+S1],end=[1,C,H1,W],strides=[1,1,1,1])
                                dict_slice[tuple12] = x12

                                dict_zero[tuple12] = False
                                if (W1-K1+S1==W):
                                    dict_zero[tuple12] = True

                                tuple21 = ((i,j),(i+1,j))
                                x21 = relay.strided_slice(Tuple[i*n+j],begin=[0,0,H1-K0+S0,0],end=[1,C,H,W1],strides=[1,1,1,1])
                                dict_slice[tuple21] = x21

                                dict_zero[tuple21] = False
                                if (H1-K0+S0==H):
                                    dict_zero[tuple21] = True

                                tuple22 = ((i,j),(i+1,j+1))
                                x22 = relay.strided_slice(Tuple[i*n+j],begin=[0,0,H1-K0+S0,W1-K1+S1],end=[1,C,H,W],strides=[1,1,1,1])
                                dict_slice[tuple22] = x22

                                dict_zero[tuple22] = False
                                if (W1-K1+S1==W or H1-K0+S0==H):
                                    dict_zero[tuple22] = True

                                if (i==0 and j==0):
                                    x2 = x1
                                if (i==0 and j!=0):
                                    if (dict_zero[((i,j-1),(i,j))]==False):
                                        c12 = dict_slice[((i,j-1),(i,j))]
                                        x2 = relay.concatenate((c12,x1),3)
                                    else:
                                        x2 = x1
                                if (i!=0 and j==0):
                                    if (dict_zero[((i-1,j),(i,j))]==False):
                                        c21 = dict_slice[((i-1,j),(i,j))]
                                        x2 = relay.concatenate((c21,x1),2)
                                    else:
                                        x2 = x1
                                if (i!=0 and j!=0):
                                    c12 = dict_slice[((i, j - 1), (i, j))]
                                    c21 = dict_slice[((i - 1, j), (i, j))]
                                    c22 = dict_slice[((i-1,j-1),(i,j))]
                                    if (dict_zero[((i,j-1),(i,j))]==False):
                                        c3 = relay.concatenate((c22,c21),3)
                                        c4 = relay.concatenate((c12,x1),3)
                                    else:
                                        c3 = c21
                                        c4 = x1
                                    if (dict_zero[((i-1,j),(i,j))]==False):
                                        x2 = relay.concatenate((c3,c4),2)
                                    else:
                                        x2 = c4

                                list_relay.append(x2)

                        #x1 = relay.strided_slice(Tuple[0],begin=[0,0,0,0],end=[1,C,H,W11],strides=[1,1,1,1])
                        #x12 = relay.strided_slice(Tuple[0],begin=[0,0,0,W11-K+S],end=[1,C,H,W1],strides=[1,1,1,1])
                        #x2 = relay.concatenate((x12,Tuple[1]),3)

                        # for i in range(n*n):
                        #     list_relay[i] = relay.device_copy(list_relay[i],ctx_cpu,ctx_gpu)

                        #x1 = relay.device_copy(x1,ctx_cpu,ctx_gpu)
                        #x2 = relay.device_copy(x2,ctx_cpu,ctx_gpu)
                        # y = relay.device_copy(call.args[1],ctx_cpu,ctx_gpu)

                        for i in range(n):
                            for j in range(n):
                                P0 = padding[0]  # top
                                P1 = padding[1]  # left
                                P2 = padding[2]  # bottom
                                P3 = padding[3]  # right
                                if (i != 0):
                                    P0 = 0
                                if (i != n - 1):
                                    P2 = 0
                                if (j != 0):
                                    P1 = 0
                                if (j != n - 1):
                                    P3 = 0
                                list_relay[i * n + j] = relay.nn.conv2d(list_relay[i * n + j], call.args[1], strides=strides,
                                                               padding=(P0, P1, P2, P3), channels=channels,
                                                               kernel_size=(K0,K1))

                        #z1 = relay.nn.conv2d(x1,y,strides=(S,S), padding=(P,P,P,0), channels=channels, kernel_size=(K,K))
                        #z2 = relay.nn.conv2d(x2,y,strides=(S,S), padding=(P,0,P,P), channels=channels, kernel_size=(K,K))

                        # for i in range(n*n):
                        #     list_relay[i] = relay.device_copy(list_relay[i],ctx_gpu,ctx_cpu)

                        #z1 = relay.device_copy(z1,ctx_gpu,ctx_cpu)
                        #z2 = relay.device_copy(z2,ctx_gpu,ctx_cpu)

                        return relay.Tuple(list_relay)
                        
                if(call.op == tvm.ir.Op.get('nn.bias_add')):
                    Tuple = self.visit(call.args[0])

                    list_relay = []
                    for i in range(n*n):
                        list_relay.append(relay.nn.bias_add(Tuple[i],call.args[1]))

                    #z1 = relay.nn.bias_add(Tuple[0],call.args[1])
                    #z2 = relay.nn.bias_add(Tuple[1],call.args[1])
                    return relay.Tuple(list_relay)
                
                if(call.op == tvm.ir.Op.get('nn.relu')):
                    Tuple = self.visit(call.args[0])

                    list_relay = []
                    for i in range(n*n):
                        list_relay.append(relay.nn.relu(Tuple[i]))

                    # z1 = relay.nn.relu(Tuple[0])
                    # z2 = relay.nn.relu(Tuple[1])
                    return relay.Tuple(list_relay)
                
                if(call.op == tvm.ir.Op.get('nn.max_pool2d')):
                    pool_size = call.attrs['pool_size']
                    strides = call.attrs['strides']
                    padding = call.attrs['padding']
                    ceil_mode = call.attrs['ceil_mode']

                    Tuple = self.visit(call.args[0])
                    x1_shape = run_infer_type(Tuple[0]).checked_type
                    C = x1_shape.concrete_shape[1]
                    H = x1_shape.concrete_shape[2]
                    Pool0 = pool_size[0]
                    Pool1 = pool_size[1]

                    list_width = []
                    list_width1 = []
                    list_height = []
                    list_height1 = []
                    sum_w = 0
                    sum_h = 0
                    for i in range(n):
                        for j in range(n):
                            shape = run_infer_type(Tuple[i*n+j]).checked_type
                            W = shape.concrete_shape[3]
                            list_width.append(W)
                            if (j == 0):
                                sum_w = 0
                            W1 = (sum_w + W) // Pool1 * Pool1 - sum_w
                            if (j == n-1):
                                W1 = W
                            list_width1.append(W1)
                            sum_w += W

                            H = shape.concrete_shape[2]
                            list_height.append(H)
                            H1 = (sum_h + H) // Pool0 * Pool0 - sum_h
                            if (i == n-1):
                                H1 = H
                            list_height1.append(H1)
                            if (j == n-1):
                                sum_h += H

                    dict_slice = {}
                    dict_zero = {}
                    list_relay = []
                    for i in range(n):
                        for j in range(n):
                            H1 = list_height1[i * n + j]
                            W1 = list_width1[i * n + j]
                            H = list_height[i * n + j]
                            W = list_width[i * n + j]
                            x1 = relay.strided_slice(Tuple[i * n + j], begin=[0, 0, 0, 0], end=[1, C, H1, W1],
                                                     strides=[1, 1, 1, 1])
                            tuple12 = ((i, j), (i, j + 1))
                            x12 = relay.strided_slice(Tuple[i * n + j], begin=[0, 0, 0, W1],
                                                      end=[1, C, H1, W], strides=[1, 1, 1, 1])
                            dict_slice[tuple12] = x12

                            dict_zero[tuple12] = False
                            if (W1 == W):
                                dict_zero[tuple12] = True

                            tuple21 = ((i, j), (i + 1, j))
                            x21 = relay.strided_slice(Tuple[i * n + j], begin=[0, 0, H1, 0],
                                                      end=[1, C, H, W1], strides=[1, 1, 1, 1])
                            dict_slice[tuple21] = x21

                            dict_zero[tuple21] = False
                            if (H1 == H):
                                dict_zero[tuple21] = True

                            tuple22 = ((i, j), (i + 1, j + 1))
                            x22 = relay.strided_slice(Tuple[i * n + j], begin=[0, 0, H1, W1],
                                                      end=[1, C, H, W], strides=[1, 1, 1, 1])
                            dict_slice[tuple22] = x22

                            dict_zero[tuple22] = False
                            if (W1 == W or H1 == H):
                                dict_zero[tuple22] = True

                            if (i == 0 and j == 0):
                                x2 = x1
                            if (i == 0 and j != 0):
                                if (dict_zero[((i, j - 1), (i, j))] == False):
                                    c12 = dict_slice[((i, j - 1), (i, j))]
                                    x2 = relay.concatenate((c12, x1), 3)
                                else:
                                    x2 = x1
                            if (i != 0 and j == 0):
                                if (dict_zero[((i - 1, j), (i, j))] == False):
                                    c21 = dict_slice[((i - 1, j), (i, j))]
                                    x2 = relay.concatenate((c21, x1), 2)
                                else:
                                    x2 = x1
                            if (i != 0 and j != 0):
                                c12 = dict_slice[((i, j - 1), (i, j))]
                                c21 = dict_slice[((i - 1, j), (i, j))]
                                c22 = dict_slice[((i - 1, j - 1), (i, j))]
                                if (dict_zero[((i, j - 1), (i, j))] == False):
                                    c3 = relay.concatenate((c22, c21), 3)
                                    c4 = relay.concatenate((c12, x1), 3)
                                else:
                                    c3 = c21
                                    c4 = x1
                                if (dict_zero[((i - 1, j), (i, j))] == False):
                                    x2 = relay.concatenate((c3, c4), 2)
                                else:
                                    x2 = c4

                            list_relay.append(x2)


                    for i in range(n*n):
                        list_relay[i] = relay.nn.max_pool2d(list_relay[i],pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)

                    return relay.Tuple(list_relay)
                    # if(x1_shape.concrete_shape[3] % 2 == 0):
                    #     z1 = relay.nn.max_pool2d(Tuple[0],pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                    #     z2 = relay.nn.max_pool2d(Tuple[1],pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                    #     return relay.Tuple([z1,z2])
                    # else:
                    #     x1 = relay.strided_slice(Tuple[1],begin=[0,0,0,0],end=[1,C,H,1],strides=[1,1,1,1])
                    #     x2 = relay.strided_slice(Tuple[1],begin=[0,0,0,1],end=[1,C,H,-1],strides=[1,1,1,1],slice_mode="size")
                    #     x1 = relay.concatenate((Tuple[0],x1),3)
                    #     z1 = relay.nn.max_pool2d(x1,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                    #     z2 = relay.nn.max_pool2d(x2,pool_size=pool_size,strides=strides,padding=padding,ceil_mode=ceil_mode)
                    #     return relay.Tuple([z1,z2])
                    
                if(call.op == tvm.ir.Op.get("nn.conv2d_transpose")):
                    channels = call.attrs['channels']
                    kernel_size = call.attrs['kernel_size']
                    padding = call.attrs['padding']
                    strides = call.attrs['strides']
                    K0 = kernel_size[0]
                    K1 = kernel_size[1]
                    #P = padding[0]
                    S0 = int(strides[0])
                    S1 = int(strides[1])
                    
                    Tuple = self.visit(call.args[0])
                    x1_shape = run_infer_type(Tuple[0]).checked_type
                    C = x1_shape.concrete_shape[1]
                    H = x1_shape.concrete_shape[2]
                    W1 = x1_shape.concrete_shape[3]

                    list_relay = []
                    list_height = []
                    list_width = []
                    for i in range(n*n):
                        #x = Tuple[i]
                        shape = run_infer_type(Tuple[i]).checked_type
                        height = shape.concrete_shape[2]
                        width = shape.concrete_shape[3]
                        list_relay.append(relay.nn.dilate(Tuple[i],strides=(1,1,S0,S1),dilation_value=0.0))
                        list_height.append(S0*(height-1)+1)
                        list_width.append(S1*(width-1)+1)

                    #x1 = relay.nn.dilate(Tuple[0],strides=(1,1,S,S),dilation_value=0.0)
                    #x2 = relay.nn.dilate(Tuple[1],strides=(1,1,S,S),dilation_value=0.0)
                    #H = S*(H-1)+1
                    #W1 = S*(W1-1)+1

                    for i in range(n):
                        for j in range(n):
                            H = list_height[i*n+j]
                            W = list_width[i*n+j]
                            Zero12 = relay.const(np.zeros((1,C,H,S0-1)))
                            Zero21 = relay.const(np.zeros((1,C,S1-1,W)))
                            Zero22 = relay.const(np.zeros((1,C,S0-1,S1-1)))

                            x1 = list_relay[i*n+j]

                            if (i!=n-1 and j==n-1):
                                x2 = relay.concatenate((x1,Zero21),2)
                            if (i==n-1 and j!=n-1):
                                x2 = relay.concatenate((x1,Zero12),3)
                            if (i!=n-1 and j!=n-1):
                                C1 = relay.concatenate((x1,Zero12),3)
                                C2 = relay.concatenate((Zero21,Zero22),3)
                                x2 = relay.concatenate((C1,C2),2)
                            if (i==n-1 and j==n-1):
                                x2 = x1

                            list_relay[i*n+j] = x2

                    #constant = relay.const(np.zeros((1,C,H,S-1)))
                    #x1 = relay.concatenate((x1,constant),3)

                    dict_slice = {}
                    dict_zero = {}
                    for i in range(n):
                        for j in range(n):
                            H = list_height[i * n + j]
                            H1 = H+S1-1
                            if (i==n-1):
                                H1 = H
                            W = list_width[i * n + j]
                            W1 = W+S0-1
                            if (j==n-1):
                                W1 = W
                            St = 1

                            x1 = list_relay[i * n + j]
                            tuple12 = ((i, j), (i, j + 1))
                            x12 = relay.strided_slice(list_relay[i * n + j], begin=[0, 0, 0, W1+St-K0],
                                                      end=[1, C, H1, W1], strides=[1, 1, 1, 1])
                            dict_slice[tuple12] = x12

                            dict_zero[tuple12] = False
                            if (W1+St-K0==W1):
                                dict_zero[tuple12] = True


                            tuple21 = ((i, j), (i + 1, j))
                            x21 = relay.strided_slice(list_relay[i * n + j], begin=[0, 0, H1+St-K1, 0],
                                                      end=[1, C, H1, W1], strides=[1, 1, 1, 1])
                            dict_slice[tuple21] = x21

                            dict_zero[tuple21] = False
                            if (H1 + St - K1 == H1):
                                dict_zero[tuple21] = True

                            tuple22 = ((i, j), (i + 1, j + 1))
                            x22 = relay.strided_slice(list_relay[i * n + j], begin=[0, 0, H1+St-K1, W1+St-K0],
                                                      end=[1, C, H1, W1], strides=[1, 1, 1, 1])
                            dict_slice[tuple22] = x22

                            dict_zero[tuple22] = False
                            if (W1 + St - K0 == W1 or H1 + St - K1 == H1):
                                dict_zero[tuple22] = True

                            if (i == 0 and j == 0):
                                x2 = x1
                            if (i == 0 and j != 0):
                                if (dict_zero[((i, j - 1), (i, j))] == False):
                                    c12 = dict_slice[((i, j - 1), (i, j))]
                                    x2 = relay.concatenate((c12, x1), 3)
                                else:
                                    x2 = x1
                            if (i != 0 and j == 0):
                                if (dict_zero[((i - 1, j), (i, j))] == False):
                                    c21 = dict_slice[((i - 1, j), (i, j))]
                                    x2 = relay.concatenate((c21, x1), 2)
                                else:
                                    x2 = x1
                            if (i != 0 and j != 0):
                                c12 = dict_slice[((i, j - 1), (i, j))]
                                c21 = dict_slice[((i - 1, j), (i, j))]
                                c22 = dict_slice[((i - 1, j - 1), (i, j))]
                                if (dict_zero[((i, j - 1), (i, j))] == False):
                                    c3 = relay.concatenate((c22, c21), 3)
                                    c4 = relay.concatenate((c12, x1), 3)
                                else:
                                    c3 = c21
                                    c4 = x1
                                if (dict_zero[((i - 1, j), (i, j))] == False):
                                    x2 = relay.concatenate((c3, c4), 2)
                                else:
                                    x2 = c4

                            #x2_shape = run_infer_type(x2).checked_type
                            list_relay[i*n+j] = x2

                    #x12 = relay.strided_slice(x1,begin=[0,0,0,W1+S-K],end=[1,C,H,W1+S-1],strides=[1,1,1,1])
                    #x2 = relay.concatenate((x12,x2),3)
                    # y = relay.device_copy(call.args[1], ctx_cpu, ctx_gpu)
                    y = relay.reverse(call.args[1],-1)
                    y = relay.reverse(y,-2)
                    y = relay.transpose(y,(1,0,2,3))

                    # for i in range(n * n):
                    #     list_relay[i] = relay.device_copy(list_relay[i], ctx_cpu, ctx_gpu)

                    #x1 = relay.device_copy(x1,ctx_cpu,ctx_gpu)
                    #x2 = relay.device_copy(x2,ctx_cpu,ctx_gpu)


                    for i in range(n):
                        for j in range(n):
                            P0 = K0-padding[0]-1  # top
                            P1 = K1-padding[1]-1  # left
                            P2 = K0-padding[2]-1  # bottom
                            P3 = K1-padding[3]-1  # right
                            if (i != 0):
                                P0 = 0
                            if (i != n - 1):
                                P2 = 0
                            if (j != 0):
                                P1 = 0
                            if (j != n - 1):
                                P3 = 0
                            list_relay[i * n + j] = relay.nn.conv2d(list_relay[i * n + j], y, strides=(1,1),
                                                                    padding=(P0, P1, P2, P3), channels=channels,
                                                                    kernel_size=(K0,K1))

                    #z1 = relay.nn.conv2d(x1,y,strides=(1,1),padding=(K-P-1,K-P-1,K-P-1,0),channels=channels,kernel_size=(K,K))
                    #z2 = relay.nn.conv2d(x2,y,strides=(1,1),padding=(K-P-1,0,K-P-1,K-P-1),channels=channels,kernel_size=(K,K))

                    # for i in range(n * n):
                    #     list_relay[i] = relay.device_copy(list_relay[i], ctx_gpu, ctx_cpu)

                    #z1 = relay.device_copy(z1,ctx_gpu,ctx_cpu)
                    #z2 = relay.device_copy(z2,ctx_gpu,ctx_cpu)
                    return relay.Tuple(list_relay)
                
                if(call.op == tvm.ir.Op.get("nn.dropout")):
                    Tuple = self.visit(call.args[0])

                    list_relay = []
                    for i in range(n*n):
                        list_relay.append(relay.nn.dropout(Tuple[i]))

                    # z1 = relay.nn.dropout(Tuple[0],call.args[1])
                    # z2 = relay.nn.dropout(Tuple[1],call.args[1])
                    return relay.Tuple(list_relay)
                        
                return call
            
            
            def visit_function(self, fn):
                n = self.n
                new_params = [self.visit(x) for x in fn.params]
                new_body = self.visit(fn.body)

                x = new_body[0*n]
                for j in range(1,n):
                    x = relay.concatenate((x,new_body[0*n+j]),3)

                for i in range(1,n):
                    y = new_body[i*n]
                    for j in range(1,n):
                        y = relay.concatenate((y,new_body[i*n+j]),3)
                    x = relay.concatenate((x,y),2)

                return relay.Function(list(new_params),x)

                #return relay.Function(list(new_params), relay.concatenate(new_body,3))

                #return relay.Function(list(new_params), new_body)
                
            def visit_tuple_getitem(self, op):
                tuple_value = self.visit(op.tuple_value)
                if not tuple_value.same_as(op.tuple_value):
                    return tuple_value
                return op
            

        return ReplaceCall().visit(func)


# f = example()
# print("before:", f)
# mod = tvm.IRModule.from_expr(f)
# seq = tvm.transform.Sequential(
#     [
#           #relay.transform.SimplifyInference(),
#           relay.transform.InferType(),
#           MyPass()
#     ]
# )
# mod1 = seq(mod)
# print("after:",mod1["main"])

# tensor = tvm.nd.array(np.random.uniform(0, 1, size=(1,3,600,600)).astype(np.float32))
# weight = tvm.nd.array(np.random.uniform(0, 1, size=(3,3,3,3)).astype(np.float32))
# bias = tvm.nd.array(np.random.uniform(0, 1, size=(3)).astype(np.float32))
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod1, target, target_host="llvm")
# module = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx_cpu,ctx_gpu))
# module.set_input("x", tensor)
# module.set_input("y", weight)
# module.set_input("bias", bias)
# module.run()
# out = module.get_output(0).asnumpy()

# result = tvm.relay.create_executor(kind="vm",mod=mod).evaluate()(tensor,weight,bias).asnumpy()
# print("optimized",out)
# print("original",result)
# print(np.all(abs(result-out) < 0.01))
