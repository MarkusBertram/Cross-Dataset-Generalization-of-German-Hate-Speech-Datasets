import h5py
import torch
from torch import nn
import sys
import torch
import torch.nn.functional as F

class KLDivWithLogits(nn.Module):

    def __init__(self):

        super(KLDivWithLogits, self).__init__()

        self.kl = nn.KLDivLoss(size_average=False, reduce=True)


    def forward(self, x, y):

        log_p = F.logsigmoid(x)
        q     = torch.sigmoid(y)

        return self.kl(log_p, q) / x.size()[0]

class ConditionalEntropy(nn.Module):

    """ estimates the conditional cross entropy of the input
    $$
    \frac{1}{n} \sum_i \sum_c p(y_i = c | x_i) \log p(y_i = c | x_i)
    $$
    By default, will assume that samples are across the first and class probabilities
    across the second dimension.
    """

    def forward(self, input):
        p     = torch.sigmoid(input)
        log_p = F.logsigmoid(input)

        H = - torch.mean(p * log_p)

        return H

class VATLoss(nn.Module):

    """ Virtual Adversarial Training Loss function
    Reference:
    TODO
    """

    def __init__(self, radius = 1): #model, radius=1):

        super(VATLoss, self).__init__()
        #self.model  = model
        self.radius = 1

        self.loss_func_nll = KLDivWithLogits()

    def forward(self, bert_output, p, model):
        # x: input features
        # p: task classifier softmax probabilities of input features x
        stacked_output = torch.stack(bert_output[2])
        x = torch.swapaxes(stacked_output, 0, 1)
        # get random vector of size x
        eps = torch.randn_like(x)
        # normalize random vector and multiply by e-6
        eps = 1e-6*F.normalize(eps)

        eps.requires_grad = True
        # calculate output of x + random vector
        eps_p = model(x + eps, input_is_bert=False)[0]
        # calculate KL divergence of output of x + random vector and output of x
        loss  = self.loss_func_nll(eps_p, p.detach())

        # calculate gradient of KL divergence
        grad = torch.autograd.grad(loss, [eps], retain_graph=True)[0]
        eps_adv = grad.detach()

        # normalize gradient
        eps_adv = F.normalize(eps_adv)#normalize_perturbation(eps_adv)
        # adversarial x is x + 1 * gradient
        x_adv = x + self.radius * eps_adv
        x_adv = x_adv.detach()

        p_adv, _ = model.forward(x_adv, input_is_bert=False)
        loss     = self.loss_func_nll(p_adv, p.detach())

        return loss

    # def _pertub(self, x, p, model, noise):
    #     eps = (torch.randn(size=x.size())).type(x.type())
        
    #     eps = 1e-6*F.normalize(eps)
    #     #eps = 1e-6 * normalize_perturbation(eps)
        
    #     eps.requires_grad = True

    #     eps_p = model(x + eps, noise = noise)[0]#self.model(x + eps)[1]

    #     loss  = self.loss_func_nll(eps_p, p.detach())
    #     loss.backward()
    #     eps_adv = eps.grad

    #     eps_adv = F.normalize(eps_adv)#normalize_perturbation(eps_adv)
    #     x_adv = x + self.radius * eps_adv

    #     return x_adv.detach()
# import contextlib
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sys

# @contextlib.contextmanager
# def _disable_tracking_bn_stats(model):

#     def switch_attr(m):
#         if hasattr(m, 'track_running_stats'):
#             m.track_running_stats ^= True
            
#     model.apply(switch_attr)
#     yield
#     model.apply(switch_attr)


# def _l2_normalize(d):
#     d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
#     d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
#     return d


# class VATLoss(nn.Module):

#     def __init__(self, xi=10.0, eps=1.0, ip=1):
#         """VAT loss
#         :param xi: hyperparameter of VAT (default: 10.0)
#         :param eps: hyperparameter of VAT (default: 1.0)
#         :param ip: iteration times of computing adv noise (default: 1)
#         """
#         super(VATLoss, self).__init__()
#         self.xi = xi
#         self.eps = eps
#         self.ip = ip

#     def forward(self, model, x, noise):
#         with torch.no_grad():
#             pred, _ = model(x, noise)
#             #pred, _ = F.softmax(model(x, noise), dim=1)
#             # pred = F.softmax(pred)

#         # prepare random unit tensor
#         d = torch.rand(x.shape).sub(0.5).to(x.device)
#         d = _l2_normalize(d)

#         with _disable_tracking_bn_stats(model):
#             # calc adversarial direction
#             for _ in range(self.ip):
#                 d.requires_grad_()
#                 pred_hat, _ = model(x + self.xi * d, noise)
#                 logp_hat = F.log_softmax(pred_hat, dim=1)
#                 adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
#                 adv_distance.backward()
#                 d = _l2_normalize(d.grad)
#                 model.zero_grad()
    
#             # calc LDS
#             r_adv = d * self.eps
#             pred_hat, _ = model(x + r_adv, noise)
#             logp_hat = F.log_softmax(pred_hat, dim=1)
#             lds = F.kl_div(logp_hat, pred, reduction='batchmean')

#         return lds

