import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds import PoincareBall
from RiemannLayers import GeodesicLayer, MobiusLayer

# Code Inspired by Emile Mathieu from Microsoft Research and Oxford Stats
# Paper: https://arxiv.org/pdf/1901.06033.pdf
# Define Encoder and Decoder Modules for the VAE


class WrappedEncoder(nn.Module):
	def __init__(self, input_dim, latent_dim, n_hlayers, hlayer_size, activation, dropout, manifold):
		super(WrappedEncoder, self).__init__()
		self.x = input_dim
		self.z = latent_dim
		self.n = n_hlayers
		self.n_size = hlayer_size
		self.activation = activation
		self.dropout = dropout
		self.manifold = manifold
		
		#Create custom encoder architechture
		layers = []
		layers.extend([nn.Linear(self.x, self.n_size), self.activation, nn.Dropout(p=self.dropout)])
		for i in range(self.n - 1):
			layers.extend([nn.Linear(self.n_size, self.n_size),
			self.activation,
			nn.Dropout(p=self.dropout)])
		self.enc = nn.Sequential(*layers)
		self.learned_param = nn.Linear(self.n_size, self.z)
		
	
	def forward(self, inputs):
		e = self.enc(inputs)
		param = self.learned_param(e)
		mu = self.manifold.expmap0(param)
		log_Sigma = F.softplus(mu)
		return mu, log_Sigma

class WrappedDecoder(nn.Module):
        def __init__(self, latent_dim, output_dim,  n_hlayers, hlayer_size, activation, dropout, manifold):
                super(WrappedDecoder, self).__init__()
                self.z = latent_dim
                self.x = output_dim
                self.n = n_hlayers
                self.n_size = hlayer_size
                self.activation = activation
                self.dropout = dropout
                self.manifold = manifold

                #Create Custom Encoder Architechture
                layers = []
                layers.extend([nn.Linear(self.z, self.n_size), self.activation, nn.Dropout(p=self.dropout)])
                for i in range(self.n - 1):
                        layers.extend([nn.Linear(self.n_size, self.n_size),
                        self.activation,
                        nn.Dropout(p=self.dropout)])
                self.dec = nn.Sequential(*layers)
                self.output_layer = nn.Linear(self.n_size, self.x)

        def forward(self, embeddings):
                emb = self.manifold.logmap0(embeddings)
                emb = self.dec(emb)
                recon = self.output_layer(emb)
                return recon

class MobiusEncoder(nn.Module):
	def __init__(self, input_dim, latent_dim, n_hlayers, hlayer_size, activation, dropout, manifold):
		super(MobiusEncoder, self).__init__()
		self.x = input_dim
		self.z = latent_dim
		self.n = n_hlayers
		self.n_size = hlayer_size
		self.activation = activation
		self.dropout = dropout
		self.manifold = manifold
	
		layers = []
		layers.extend([nn.Linear(self.x, self.n_size), self.activation, nn.Dropout(p=self.dropout)])
		for i in range(self.n - 1):
			layers.extend([nn.Linear(self.n_size, self.n_size),
			self.activation,
			nn.Dropout(p=self.dropout)])
		self.enc = nn.Sequential(*layers)
		self.sigma_out = nn.Linear(self.n_size, self.z)
		self.output_layer = MobiusLayer(self.n_size, self.z, self.manifold)

	def forward(self, inputs):
		e = self.enc(inputs)
		mu = self.output_layer(e)
		mu = self.manifold.expmap0(mu)
		log_Sigma = F.softplus(self.sigma_out(e))
		return mu, log_Sigma


class GeodesicDecoder(nn.Module):
	def __init__(self, latent_dim, output_dim, n_hlayers, hlayer_size, activation, dropout, manifold):
		super(GeodesicDecoder, self).__init__()
		self.z = latent_dim
		self.x = output_dim
		self.n = n_hlayers
		self.n_size = hlayer_size
		self.activation = activation
		self.dropout = dropout
		self.manifold = manifold
		
	
		input_layer = GeodesicLayer(self.z, self.n_size, self.manifold)
		layers = [input_layer]
		layers.extend([self.activation, nn.Dropout(p=self.dropout)])
		for i in range(self.n - 1):
			layers.extend([nn.Linear(self.n_size, self.n_size),
			self.activation,
			nn.Dropout(p=self.dropout)])
		self.dec = nn.Sequential(*layers)
		self.output_layer = nn.Linear(self.n_size, self.x)
	
	def forward(self, embeddings):
		decode = self.dec(embeddings)
		recon = self.output_layer(decode)
		return recon
	
class MobiusDecoder(nn.Module):
	def __init__(self, latent_dim, output_dim, n_hlayers, hlayer_size, activation, dropout, manifold):
		super(MobiusDecoder, self).__init__()
		self.z = latent_dim
		self.x = output_dim
		self.n = n_hlayers
		self.n_size = hlayer_size
		self.activation = activation
		self.dropout = dropout
		self.manifold = manifold
		

		layers = []
		layers.extend([MobiusLayer(self.z, self.n_size, self.manifold), self.activation, nn.Dropout(p=self.dropout)])
		for i in range(self.n - 1):
			layers.extend([nn.Linear(self.n_size, self.n_size),
			self.activation,
			nn.Dropout(p=self.dropout)])
		self.dec = nn.Sequential(*layers)
		self.output_layer = nn.Linear(self.n_size, self.x)

	def forward(self, embeddings):
		emb = self.dec(embeddings)
		recon = self.output_layer(emb)
		return recon
	
#Debugging Code
#if __name__ == '__main__':
#	inputs = torch.randn(64, 6000).double()
#	x = inputs.shape[1]
#	z = 2
#	n = 2
#	n_size = 256
#	activ = nn.LeakyReLU()
#	drop_rate = 0.2
#	manifold = PoincareBall(c=1)
#	Encoders = [WrappedEncoder(x, z, n, n_size, activ, drop_rate, manifold),
#	MobiusEncoder(x, z, n, n_size, activ, drop_rate, manifold)]
#	for e in Encoders:
#		e = e.double()
#		print(e(inputs))
#	embeddings = torch.randn(64, 2).double()
#	Decoders = [GeodesicDecoder(z, x, n, n_size, activ, drop_rate, manifold),
#	MobiusDecoder(z, x, n, n_size, activ, drop_rate, manifold)]
#	for d in Decoders:
#		d = d.double()
#		print(d(embeddings))
