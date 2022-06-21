import numpy as np
from numpy import ndarray
from collections import defaultdict
import math

class Adam(object):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0, amsgrad=False):
        
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.state = defaultdict(dict)
    
    def step(self, params, grads):
        params = [p.asnumpy() for n, p in params.items()]
        grads.reverse()

        params_with_grad = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = self.betas
        
        for i, p in enumerate(params):
            if grads[i] is not None:
                params_with_grad.append(p)
                
            state = self.state[i]
            # lazy state initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = np.zeros_like(p)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = np.zeros_like(p)
                if self.amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = np.zeros_like(p)

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])

            if self.amsgrad:
                max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            # update the steps for each param group update
            state['step'] += 1
            # record the step after step update
            state_steps.append(state['step'])
        
        updated_params = self.adam(params_with_grad,
                                grads,
                                exp_avgs,
                                exp_avg_sqs,
                                max_exp_avg_sqs,
                                state_steps,
                                self.amsgrad,
                                beta1,
                                beta2,
                                self.lr,
                                self.weight_decay,
                                self.eps)
        
        for i in range(len(self.state)):
            self.state[i]['exp_avg'] = exp_avgs[i]
            self.state[i]['exp_avg_sq'] = exp_avg_sqs[i]
        
        return updated_params
    
    def adam(self,
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad,
            beta1,
            beta2,
            lr,
            weight_decay,
            eps):
        
        for i, param in enumerate(params):
            grad = grads[i].asnumpy()
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg = exp_avg * beta1 + grad * (1 - beta1)
            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                # torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                max_exp_avg_sqs[i] = np.maximum(max_exp_avg_sqs[i], exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                # denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                denom = (np.sqrt(max_exp_avg_sqs[i]) / math.sqrt(bias_correction2)) + eps
            else:
                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                denom = (np.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps

            step_size = lr / bias_correction1

            params[i] = param + (-step_size) * (exp_avg / denom)
            
            exp_avgs[i] = exp_avg
            exp_avg_sqs[i] = exp_avg_sq
        
        return params