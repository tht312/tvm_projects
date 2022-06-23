#!/usr/bin/env python
# coding: utf-8

# In[174]:


import inspect
import types
import sys

import torch
from torch.utils import dlpack

import tvm
from tvm import relay
from tvm.contrib import graph_runtime

import numpy

def work_on_fn(pass_cls):
    def apply_pass(fn_or_mod):
        if isinstance(fn_or_mod, tvm.IRModule):
            return pass_cls()(fn_or_mod)
        if isinstance(fn_or_mod, tvm.relay.Function):
            return pass_cls()(
                       tvm.IRModule({'main': fn_or_mod}))['main']
        raise NotImplemented("unsupporded type {}".format(type(fn_or_mod)))
    return apply_pass

infer_type = work_on_fn(tvm.relay.transform.InferType)
to_graph_normal_form = work_on_fn(tvm.relay.transform.ToGraphNormalForm)
dead_code_elimination = work_on_fn(tvm.relay.transform.DeadCodeElimination)
eliminate_common_subexpr = work_on_fn(tvm.relay.transform.EliminateCommonSubexpr)

class ShapeConstDedupMutator(tvm.relay.ExprMutator):
    def __init__(self):
        super().__init__()
        self.shape_consts = {}

    def visit_call(self, call):
        if (isinstance(call.op, tvm.ir.Op) 
            and call.op.name in {"reshape", "broadcast_to", "collapse_sum_to"}
            and isinstance(call.args[1], tvm.relay.Constant)):
            # assert list(call.attrs.newshape) == list(call.args[1].data.asnumpy())
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            const = new_args[1]
            assert const.data.dtype.startswith('int') and len(const.data.shape)==1
            key = tuple(const.data.asnumpy())
            if key in self.shape_consts:
                new_args[1] = self.shape_consts[key]
            else:
                self.shape_consts[key] = new_args[1]
            return tvm.relay.Call(new_fn, new_args, call.attrs)
        return super().visit_call(call)


class TransposeDedupMutator(tvm.relay.ExprMutator):
    def visit_call(self, call):
        if (isinstance(call.op, tvm.ir.Op) and call.op.name == "transpose"
            and isinstance(call.args[0], tvm.relay.Call) 
            and isinstance(call.args[0].op, tvm.ir.Op) and call.args[0].op.name == "transpose"):
            axes = [call.args[0].attrs.axes[int(i)] for i in call.attrs.axes]
            new_inp = self.visit(call.args[0].args[0])
            if axes == list(range(len(axes))): # neutral permutation, should really do this separately...
                return new_inp
            return tvm.relay.transpose(new_inp, axes)
        return super().visit_call(call)

#@tvm.relay.transform.function_pass(opt_level=1)
#def TransposeDedup(fn, mod, ctx):
#    return TransposeDedupMutator().visit(fn)

class ZeroZapp(tvm.relay.dataflow_pattern.DFPatternCallback):
    def __init__(self):
        self.zeros = tvm.relay.dataflow_pattern.is_op("zeros")() | tvm.relay.dataflow_pattern.is_constant()
        self.other_tensor = tvm.relay.dataflow_pattern.wildcard()
        self.pattern = (tvm.relay.dataflow_pattern.is_op("add")(self.zeros, self.other_tensor)) | (tvm.relay.dataflow_pattern.is_op("add")(self.other_tensor, self.zeros))
        self.require_type = True
        self.rewrite_once = True


    def callback(self, pre, post, node_map):
        rt = node_map[self.pattern][0]
        zeros = node_map[self.zeros][0]
        ot = node_map[self.other_tensor][0]
        if isinstance(zeros, tvm.relay.Constant):
            val = zeros.data.asnumpy()
            if not ((val == 0) if numpy.isscalar(val) else (val == 0).all()):
                return rt
        # I don't know why I don't reliably get checked types here...
        if (((rt._checked_type_ is not None) and (ot._checked_type_ == rt._checked_type_))
            or (rt.type_args[0] == rt.type_args[1])):
            return ot
        elif (rt._checked_type_ is not None):
            return tvm.relay.broadcast_to(ot, list(rt._checked_type_.shape))
        return rt

class OneZapp(tvm.relay.dataflow_pattern.DFPatternCallback):
    def __init__(self):
        self.ones = tvm.relay.dataflow_pattern.is_op("ones")() | tvm.relay.dataflow_pattern.is_constant()
        self.other_tensor = tvm.relay.dataflow_pattern.wildcard()
        self.pattern = (tvm.relay.dataflow_pattern.is_op("multiply")(self.ones, self.other_tensor)) | (tvm.relay.dataflow_pattern.is_op("multiply")(self.other_tensor, self.ones))
        self.require_type = True
        self.rewrite_once = True

    def callback(self, pre, post, node_map):
        global val
        rt = node_map[self.pattern][0]
        ones = node_map[self.ones][0]
        ot = node_map[self.other_tensor][0]
        if isinstance(ones, tvm.relay.Constant):
            val = ones.data.asnumpy()
            if not ((val == 1) if numpy.isscalar(val) else (val == 1).all()):
                return rt
        if (((rt._checked_type_ is not None) and (ot._checked_type_ == rt._checked_type_))
            or (rt.type_args[0] == rt.type_args[1])):
            return ot
        if (rt._checked_type_ is not None):
            return tvm.relay.broadcast_to(ot, list(rt._checked_type_.shape))
        return rt


class LikeZapp(tvm.relay.dataflow_pattern.DFPatternCallback):
    def __init__(self):
        self.translations_with_dt = {'zeros_like': tvm.relay.zeros,
                                     'ones_like': tvm.relay.ones}
        self.data_tensor = tvm.relay.dataflow_pattern.wildcard()
        self.pattern_tensor = tvm.relay.dataflow_pattern.wildcard()
        self.pattern = ((tvm.relay.dataflow_pattern.is_op("zeros_like")
                        | tvm.relay.dataflow_pattern.is_op("ones_like")
                        )(self.data_tensor)
                        ) | ((
                        tvm.relay.dataflow_pattern.is_op("collapse_sum_like")
                        | tvm.relay.dataflow_pattern.is_op("reshape_like")
                        | tvm.relay.dataflow_pattern.is_op("broadcast_to_like")
                       )(self.data_tensor, self.pattern_tensor))
        self.require_type = True
        self.rewrite_once = True

    def callback(self, pre, post, node_map):
        data = node_map[self.data_tensor][0]
        res = node_map[self.pattern][0]
        if res.op.name in self.translations_with_dt:
            ret = self.translations_with_dt[res.op.name](list(res.type_args[0].shape),
                                                              res.type_args[0].dtype) # which dtype?
            return ret
        if (res.type_args[0] is not None and res.type_args[0] == res.type_args[1]):
            return data
        if res.op.name == 'broadcast_to_like':
            return tvm.relay.broadcast_to(data, list(res.type_args[1].shape))
        if res.op.name == 'reshape_like':
            return tvm.relay.reshape(data, list(res.type_args[1].shape))
        if res.op.name == 'collapse_sum_like':
            return tvm.relay.collapse_sum_to(data, list(res.type_args[1].shape))
        return res

class ExternalizeDropout(tvm.relay.dataflow_pattern.DFPatternCallback):
    # TVM doesn't have a Dropout defined (for inference it can be deleted)
    # but it also does not appear to have random, so we make the random draw
    # an input
    def __init__(self):
        self.dropout_info = {}
        self.counter = 0
        self.inp = tvm.relay.dataflow_pattern.wildcard()
        self.dropout = tvm.relay.dataflow_pattern.is_op("nn.dropout")(self.inp)
        self.pattern = tvm.relay.dataflow_pattern.is_tuple_get_item(self.dropout, 0)
        self.require_type = True
        self.rewrite_once = True

    def callback(self, pre, post, node_map):
        res = node_map[self.pattern][0]
        dropout = node_map[self.dropout][0]
        inp = node_map[self.inp][0]
        typ = dropout.type_args[0]
        rate = dropout.attrs.rate
        name = f"dropout:{self.counter}"
        self.counter += 1
        do_var = tvm.relay.var(name, type_annotation=typ)
        # do_var = tvm.relay.device_copy(do_var, "cpu", "cuda")
        self.dropout_info[name] = (rate, typ)
        return inp * (do_var * tvm.relay.const(1 / (1 - rate), dtype=typ.dtype))

def externalize_dropout(fn):
    edo = ExternalizeDropout()
    fn = tvm.relay.dataflow_pattern.rewrite(edo, fn)
    fn = tvm.relay.Function(list(fn.params) + list(tvm.relay.analysis.free_vars(fn)),
                              fn.body)
    return fn, edo.dropout_info

def tensor_to_tvm(t):
    return tvm.nd.from_dlpack(dlpack.to_dlpack(t))

def tensor_from_tvm(a):
    return (dlpack.from_dlpack(a.to_dlpack()))

def create_function(mod):
    # the converter will output arguments in an arbitrary order (well, by position of use), we want that of the input
    fn = mod['main']
    
    # fn = tvm.relay.Function([fn.params[tmp_arg_idx[n]] for n in arg_order], fn.body)
    
    fn = TransposeDedupMutator().visit(fn)

    # prepare function to also use grad_out
    fn = infer_type(fn)
    output_type = fn.body.checked_type # fn.ret_type :)
    
    if isinstance(output_type, tvm.relay.TensorType):
        gr_out = tvm.relay.var("gr:out:0", output_type)
        fn_for_gr = tvm.relay.Function(list(fn.params) + [gr_out], tvm.relay.sum(fn.body * gr_out))
    else:
        # we can try to handle tuples of tensors, but our nesting patience ends there
        assert (isinstance(output_type, tvm.relay.TupleType) and
                all([isinstance(f, tvm.relay.TensorType) for f in output_type.fields]))
        gr_outs = [tvm.relay.var(f"gr:out:{i}", t) for i, t in enumerate(output_type.fields)]
        prods_with_gr_out = [tvm.relay.sum(tvm.relay.TupleGetItem(fn.body, i) * go_i)
                             for i, go_i in enumerate(gr_outs)]
        s = prods_with_gr_out[0]
        for p in prods_with_gr_out[1:]:
            s = s + p
        fn_for_gr = tvm.relay.Function(list(fn.params) + gr_outs, s)
    
    fn_for_gr = infer_type(fn_for_gr)
    fn_for_gr, dropout_info = externalize_dropout(fn_for_gr)
    fn_for_gr = infer_type(fn_for_gr)
    
    # take the gradient
    grfn = tvm.relay.transform.gradient(fn_for_gr, mode='first_order')
    grfn = to_graph_normal_form(grfn)
    
    # removing of unneeded outputs and simplifications of the gradient

    # Now we have (sum(orig_out * grad_out), (grad_inp_1, ..., grad_inp_n, grad_grad_out, gr_dropout ...))
    # but we only want orig_out and grad_inp_1, ..., grad_inp_n
    def is_aux_input(p):
        return p.name_hint.startswith('dropout') or p.name_hint.startswith('gr:out')

    # the gr_out and dropout parameters will have gradients computed, but we do not want that
    grads_to_keep = tvm.relay.Tuple([g for p, g in zip(grfn.params, grfn.body.fields[1].fields)
                                       if not is_aux_input(p)])

    assert grfn.body.fields[0].op.name == 'sum'
    assert grfn.body.fields[0].args[0].op.name == 'multiply'
    if isinstance(output_type, tvm.relay.TensorType):
        orig_out = grfn.body.fields[0].args[0].args[0]
    else:
        assert isinstance(output_type, tvm.relay.TupleType)
        orig_out = grfn.body.fields[0].args[0].args[0].tuple_value
    
    out_and_grad = tvm.relay.Tuple([orig_out, grads_to_keep])
    out_and_grad_fn = tvm.relay.Function(grfn.params, out_and_grad)
    out_and_grad_fn = infer_type(out_and_grad_fn)
    out_and_grad_fn = dead_code_elimination(out_and_grad_fn)
    out_and_grad_fn = eliminate_common_subexpr(out_and_grad_fn)
    out_and_grad_fn = infer_type(out_and_grad_fn)
    out_and_grad_fn = tvm.relay.dataflow_pattern.rewrite(LikeZapp(), out_and_grad_fn)
    out_and_grad_fn = infer_type(out_and_grad_fn)
    out_and_grad_fn = tvm.relay.dataflow_pattern.rewrite(ZeroZapp(), out_and_grad_fn)
    out_and_grad_fn = infer_type(out_and_grad_fn)
    out_and_grad_fn = tvm.relay.dataflow_pattern.rewrite(OneZapp(), out_and_grad_fn)
    out_and_grad_fn = infer_type(out_and_grad_fn)
    out_and_grad_fn = tvm.relay.dataflow_pattern.rewrite(OneZapp(), out_and_grad_fn)
    out_and_grad_fn = infer_type(out_and_grad_fn)
    out_and_grad_fn = dead_code_elimination(out_and_grad_fn)
    out_and_grad_fn = eliminate_common_subexpr(out_and_grad_fn)
    out_and_grad_fn = infer_type(out_and_grad_fn)
    
    out_and_grad_mod = tvm.IRModule({"main": out_and_grad_fn})

    return {'fw_mod': mod, 'fw_and_bw_mod': out_and_grad_mod, 'drop_info': dropout_info, "out_type": output_type}