import torch 
import torch.nn as nn
import torch.functional as F
import geoopt
from geoopt.manifolds import PoincareBall
from Modules import *
from WrappedNormal import *
from torch.distributions import Normal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#Code My Hyperbolic Variational Autoencoder

class HVAE(nn.Module):
	def __init__(self,enc, dec, prior, posterior, likelihood, **kwargs):
		super(HVAE, self).__init__()
		self.enc = enc
		self.dec = dec
		self.prior = prior
		self.posterior = posterior
		self.likelihood = likelihood
		self.c = kwargs.get('c', 1.0)
		self.learn_prior_logvar = kwargs.get('learn_logvar', False)
		self.latent_mean = nn.Parameter(torch.zeros(1, self.enc.z), requires_grad = False)
		self.latent_logvar = nn.Parameter(torch.zeros(1,1), requires_grad=self.learn_prior_logvar)
	
	@property
	def latent_param(self):
		return self.latent_mean.mul(1), F.softplus(self.latent_logvar).div(math.log(2)).mul(1)


	def forward(self, inputs, K=1):
		#It should be mentioned that the K of the model here is assumed 1
		mu, Sigma = self.enc(inputs)
		encoder_dist = self.posterior(mu, Sigma, self.c, self.enc.z)
		latent_samples = encoder_dist.rsample(torch.Size([K]))
		decoder_dist_mean = self.dec(latent_samples)
		decoder_dist = self.likelihood(decoder_dist_mean, torch.ones_like(decoder_dist_mean))
		return encoder_dist, decoder_dist, latent_samples 
	
	def generate(self, N, K):
		self.eval()
		with torch.no_grad():
			latent_mean = self.latent_param[0]
			recon_means = self.dec(latent_mean)
			decoder_dist_mean = self.dec(
				self.prior(*self.latent_param, self.c, self.enc.z).sample(torch.Size([N])))
			samples = self.likelihood(decoder_dist_mean, torch.ones_like(decoder_dist_mean)).sample(torch.Size([K]))
		return latent_mean, recon_means, samples
	
	def reconstruct(self, inputs):
		self.eval()
		with torch.no_grad():
			posterior = self.posterior(*self.enc(inputs), self.c, self.enc.z)
			decoder_dist_mean = self.dec(posterior.rsample(torch.Size([1])).squeeze(0))
		return decoder_dist_mean

	def to_latent(self, inputs):
		self.eval()
		with torch.no_grad():
			mu, Sigma = self.enc(inputs)
			latent_samples = self.posterior(mu, Sigma, self.c, self.enc.z).rsample([inputs.shape[0], self.enc.z])
		return latent_samples
	
	def plot_2D_embeddings(self, inputs, title='PVAE Latent Space',fig_out='./latent_emb.pdf', zoom_out = './zoom_emb.pdf', labels=None):
		latent_emb = self.to_latent(inputs)
		latent_emb = pd.DataFrame(latent_emb, columns =("dim1", "dim2"))
		colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B',
                  '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                  '#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B',
                  '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
                  '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
                  '#CCEBC5', '#FFED6F']
		
		fig = plt.figure(figsize = (9.5, 9))
		ax = plt.gca()
		circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
		ax.add_patch(circle)
		ax.plot(0, 0, 'x', c=(0, 0, 0), ms=4)
		if not (labels is None):
			latent_emb['labels'] = labels
			for i in set(labels):
				subset = latent_emb[latent_emb['labels'] == i]
				plt.scatter(subset['dim1'], subset['dim2'], label = i)  
				ax.legend(fontsize=12)
		else:
			plt.scatter(latent_emb['dim1'], latent_emb['dim2'])
		plt.title(title)
		plt.xlabel('PVAE_1')
		plt.ylabel('PVAE_2')
		#plt.show()
		plt.savefig(fig_out, dpi = 600)
		
		#Create Zoom
		fig.clear()
		latent_emb = np.transpose(self.to_latent(inputs).cpu().numpy())
		sqnorm = np.sum(latent_emb ** 2, axis = 1, keepdims=True)
		print(sqnorm)
		arcos_theta  = 1 + 2 * sqnorm / (sqnorm - 1)
		dist = np.arccosh(arcos_theta)
		dist = np.sqrt(dist)
		dist /= dist.max()
		sqnorm[sqnorm == 0] = 1
		latent_emb = dist * latent_emb / np.sqrt(sqnorm)
		latent_emb = np.transpose(latent_emb)
		latent_emb = pd.DataFrame(latent_emb, columns =("dim1", "dim2"))

		fig = plt.figure(figsize = (9.5, 9))
		ax = plt.gca()
		circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
		ax.add_patch(circle)
		if not (labels is None):
			latent_emb['labels'] = labels
			for i in set(labels):
				subset = latent_emb[latent_emb['labels'] == i]
				plt.scatter(subset['dim1'], subset['dim2'], label = i)
				ax.legend(fontsize=12)
		else:
			plt.scatter(latent_emb['dim1'], latent_emb['dim2'])
		plt.title(title)
		plt.xlabel('PVAE_1')
		plt.ylabel('PVAE_2')
		#plt.show()
		plt.savefig(zoom_out, dpi = 600)
		

#Debug Code
#Comment Out
#if __name__ == '__main__':
#	inputs = torch.rand(64, 6000).double()
#	x = inputs.shape[1]
#	z = 2
#	n = 2
#	n_size = 256
#	activ = nn.LeakyReLU()
#	drop_rate = 0.2
#	manifold = PoincareBall(c=1.0)
#	encoder = MobiusEncoder(x, z, n, n_size, activ, drop_rate, manifold)
#	decoder = MobiusDecoder(z, x, n, n_size, activ, drop_rate, manifold)
#	prior = WrappedNormal
#	posterior = WrappedNormal
#	likelihood = Normal
#	
#	model = HVAE(encoder, decoder, 
#	prior, posterior, likelihood).double()
#	print(model.forward(inputs))
#	print(model.generate(2, 100))
#	print(model.reconstruct(inputs))
#	print(model.to_latent(inputs))
#	print(model.plot_2D_embeddings(inputs))
