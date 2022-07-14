# import h5py
# import torch
# from torch import nn

# import torch
# import torch.nn.functional as F

# class KLDivWithLogits(nn.Module):

#     def __init__(self):

#         super(KLDivWithLogits, self).__init__()

#         self.kl = nn.KLDivLoss(size_average=False, reduce=True)
#         self.logsoftmax = nn.LogSoftmax(dim = 1)
#         self.softmax = nn.Softmax(dim = 1)


#     def forward(self, x, y):

#         log_p = self.logsoftmax(x)
#         q     = self.softmax(y)

#         return self.kl(log_p, q) / x.size()[0]


# def normalize_perturbation(d):
#     d_ = d.view(d.size()[0], -1)
#     eps = d.new_tensor(1e-12)
#     output = d / torch.sqrt(torch.max((d_**2).sum(dim = -1), eps)[0] )
#     return output

# class VATLoss(nn.Module):

#     """ Virtual Adversarial Training Loss function
#     Reference:
#     TODO
#     """

#     def __init__(self, model, radius=1):

#         super(VATLoss, self).__init__()
#         self.model  = model
#         self.radius = 1

#         self.loss_func_nll = KLDivWithLogits()

#     def forward(self, x, p):

#         x_adv    = self._pertub(x, p)
#         _, p_adv = self.model.forward(x_adv,noise=True,training=True)
#         loss     = self.loss_func_nll(p_adv, p.detach())

#         return loss

#     def _pertub(self, x, p):
#         eps = (torch.randn(size=x.size())).type(x.type())

#         eps = 1e-6 * normalize_perturbation(eps)
#         eps.requires_grad = True

#         eps_p = self.model(x + eps)[1]

#         loss  = self.loss_func_nll(eps_p, p.detach())
#         loss.backward()
#         eps_adv = eps.grad

#         eps_adv = normalize_perturbation(eps_adv)
#         x_adv = x + self.radius * eps_adv

#         return x_adv.detach()
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, noise):
        with torch.no_grad():
            pred, _ = model(x, noise)
            #pred, _ = F.softmax(model(x, noise), dim=1)
            # pred = F.softmax(pred)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat, _ = model(x + self.xi * d, noise)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat, _ = model(x + r_adv, noise)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

class ConditionalEntropy(nn.Module):

    """ estimates the conditional cross entropy of the input
    $$
    \frac{1}{n} \sum_i \sum_c p(y_i = c | x_i) \log p(y_i = c | x_i)
    $$
    By default, will assume that samples are across the first and class probabilities
    across the second dimension.
    """

    def forward(self, input):
        p     = F.softmax(input, dim=1)
        log_p = F.log_softmax(input, dim=1)

        H = - (p * log_p).sum(dim=1).mean(dim=0)

        return H