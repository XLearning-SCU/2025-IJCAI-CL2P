import torch

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    # Copyright (c) Facebook, Inc. and its affiliates.
    # All rights reserved.

    # This source code is licensed under the license found in the
    # LICENSE file in the root directory of this source tree.

    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

    
def get_optimizer(params, lr=1e-3, op_name='adam'):

    if op_name == 'sgd':
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False, weight_decay=0.0001)
    elif op_name == 'adam':
        opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0001, betas=[0.9, 0.999])
    elif op_name == 'adamw':
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0001, betas=[0.9, 0.95])
    elif op_name == 'lars':
        opt = LARS(params, lr=lr, weight_decay=0.0001, momentum=0.9)
    else:
        raise ValueError('optimizer must be sgd or adam.')
    return opt



