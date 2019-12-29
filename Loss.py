import torch
import torch.nn as nn
import geoopt
from WrappedNormal import WrappedNormal
from torch.distributions import Normal
from VAE import HVAE
from Modules import *
import math
from numpy import prod
def mse_loss(inputs, targets):
    return torch.sum((targets - inputs) ** 2)

def hyp_loss(model, inputs, K = 1, beta = 1.0):
    enc_dist, dec_dist, latent_samples = model(inputs)
    lnPosterior = dec_dist.log_prob(inputs).sum(-1)
    latent_dist = model.prior(*model.latent_param, model.c, model.enc.z)
    kld = enc_dist.log_prob(latent_samples).sum(-1) - latent_dist.log_prob(latent_samples).sum(-1)
    mse = mse_loss(model.reconstruct(inputs), inputs)
    loss = -lnPosterior.mean(0).sum() + beta * kld.mean(0).sum() + K * mse
    return loss

def iwae_vec(model, inputs, K):
    enc_dist, dec_dist, latent_samples = model(inputs)
    lpz = model.prior(*model.latent_param, model.c, model.enc.z).log_prob(latent_samples).sum(-1)
    lpx_z = dec_dist.log_prob(inputs).sum(-1)
    lqz_x = enc_dist.log_prob(latent_samples).sum(-1)
    obj = lpz + lpx_z - lqz_x
    m, _ = torch.max(obj, dim=0, keepdim = True)
    value0 = obj - m
    m = m.squeeze(0)
    log_sum_exp = m + torch.log(torch.sum(torch.exp(value0), dim = 0, keepdim = False))
    log_mean_exp = log_sum_exp - math.log(obj.size(0))
    return -log_mean_exp.sum()

#def iwae_loss(model, inputs, K):
#    split_size = int(inputs.size(0) / K * prod(inputs.size()) / (3e7))
#    print(split_size)
#    if split_size >= inputs.size(0):
#        obj =  iwae_vec(model, inputs)
#    else:
#        obj = 0
#        for bx in inputs.split(split_size):
#            obj = obj + iwae_vec(model, bx)
    return obj

	         
# Debug Code
# Comment Out
#if __name__ == '__main__':
#    inputs = torch.rand(64, 6000).double()
#    x = inputs.shape[1]
#    z = 2
#    n = 2
#    n_size = 256
#    activ = nn.LeakyReLU()
#    drop_rate = 0.2
#    manifold = PoincareBall(c=1.0)
#    encoder = MobiusEncoder(x, z, n, n_size, activ, drop_rate, manifold)
#    decoder = MobiusDecoder(z, x, n, n_size, activ, drop_rate, manifold)
#    prior = WrappedNormal
#    posterior = WrappedNormal
#    likelihood = Normal
#    model = HVAE(encoder, decoder, 
#    prior, posterior, likelihood).double()
#    print(hyp_loss(model, inputs))
#    print(iwae_vec(model, inputs, 5000))
