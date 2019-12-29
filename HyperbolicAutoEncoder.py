############################################
# Hyperbolic Variational Autoencoder Code  #
############################################

"""
He I will create the base code for a variational autoencoder embedding for my analysis of the TCDD data.
This code will eventually be broken up into several documents depending on what makes sense.

My Goals are:
Code the poincare embedding
Code the hyperbolic distributions from the emile mathieu PVAE project
Code @ https://github.com/emilemathieu/pvae/ 
Code the variational autoencoder architechture
Code ODE's on the latent space
"""

# Importing modules

import torch
import torch.nn as nn
import geoopt
from geoopt.manifolds.poincare.math import _lambda_x, arsinh, tanh
from torch.distributions.distribution import Distribution
import torch.distributions.constraints as constraint
from torch.distributions.utils import _standard_normal, broadcast_all
import matplotlib.pyplot as plt
import pandas as pd

#Define Normal Distribution on the poincare ball manifold using an exponnetial map
class wrapped_hyper_norm(Distribution):
		
	arg_constraints = {'mu':constraint.positive, 'log_var': constrain.real}
 
	@property
	def mean(self):
		return self.mu

	@property
	def log_var(self):
		return self.log_var

	@property
	def batch_size(self):
		return self.size
	
	@property
	def stddev(self):
		return NotImplementedError
	
	def __init__(mu, log_var, c, **kwargs):
                self.mu = mu
                self.log_var = log_var
                self.manifold = geoopt.manifold.PoincareBall(c)
		self.batch_size = mu.size[0]
		self.event_size = mu.size[1]
		self.sample_size = inputs.shape
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		seld.curv = torch.tensor(c).to(self.device)
		self.zero = torch.zeros(1, self.event_size).to(self.device)
		super(wrapped_hyper_norm, self).__init__(self.batch_size, self.event_size)
 		
	def sample(self):
		with torch.no_grad():
			self.rsample(self.size)
	
	def rsample(self, sample_size = self.sample_size):
		v = self.log_var *  _standard_normal(sample_size, dtype=self.mu.dtype, device=self.device)
		u = v /_lambda_x(self.zero)
		u = self.manifold.transp(self.zero, self.mu, v)
		z = self.manifold.expmap(self.mu, u)
		return z

	def log_prob(self, x):
        	shape = x.shape
        	mu = self.mu.unsqueeze(0).expand(x.shape[0], * self.batch_shape, self.event_shape)
        	if len(shape) < len(mu.shape): 
			x = x.unsqueeze(1)
        	v = self.manifold.logmap(mu, x)
        	v = self.manifold.transp(mu, self.zero, v)
        	u = v * self.manifold.lambda_x(self.zero, keepdim=True)
        	norm_pdf = Normal(torch.zeros_like(self.log_var), self.log_var).log_prob(u).sum(-1, keepdim=True)
        	d = self.manifold.norm(mu, x, keepdim=True) if is_vector else self.manifold.dist(mu, x, keepdim=True)
		logdetexp = (self.event_size - 1) * (torch.sinh(self.curv.sqrt()*d) / self.curv.sqrt() / d).log()
        	result = norm_pdf - logdetexp
        	return result

#Define the Reimann Layer for different distributions
class RiemannianLayer(nn.Module):
	def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
		super(RiemannianLayer, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.manifold = manifold

		self._weight = Parameter(torch.Tensor(out_features, in_features))
		self.over_param = over_param
		self.weight_norm = weight_norm
		if self.over_param:
			self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
		else:
			self._bias = Parameter(torch.Tensor(out_features, 1))
		self.reset_parameters()

		@property
		def weight(self):
			return self.manifold.transp0(self.bias, self._weight) # weight \in T_0 => weight \in T_bias

		@property
		def bias(self):
			if self.over_param:
			return self._bias
		else:
			return self.manifold.expmap0(self._weight * self._bias) # reparameterisation of a point on the manifold

	def reset_parameters(self):
		init.kaiming_normal_(self._weight, a=math.sqrt(5))
		fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
		bound = 4 / math.sqrt(fan_in)
		init.uniform_(self._bias, -bound, bound)
		if self.over_param:
			with torch.no_grad(): self._bias.set_(self.manifold.expmap0(self._bias))

# Define the geodesic layer for  stronger results
class GeodesicLayer(RiemannianLayer):
	def __init__(self, in_features, out_features, c, over_param=False, weight_norm=False):
        	self.curv = c
		self.manifold = geoopt.manifolds.PoincareBall(c=c)
		super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)
	
	def normdist2plane(self, inputs, norm, keepdim=False, signed=True):
		x = inputs 
		a = self.bias 
		p = self.weight
		dim = x.shape[1]
		c = self.curv
		sqrt_c = c ** 0.5
		diff = self.manifold.mobius_add(-p, x, dim=dim)
		diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(1e-15)
		sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
		if not signed:
			sc_diff_a = sc_diff_a.abs()
		a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(1e-15)
		num = 2 * sqrt_c * sc_diff_a
		denom = (1 - c * diff_norm2) * a_norm
		res = arsinh(num / denom.clamp_min(1e-15)) / sqrt_c
		if norm:
			res = res * a_norm
		return res
		
	def forward(self, inputs):
		res = self.normdist2plane(inputs, self.weight_norm)
		return res
		
#Define Model Architechture
class HAE(nn.Module):
	def __init__(x_dim, z_dim, n_dim,  **kwargs):
		"""
		x_dim: input dimension
		z_dim: embedding (latent space) dimension
		n_dim: dimensionality of hidden layer
		c: how much negative curvature (0, inf)
		 
		"""
		self.x = x_dim
		self.z = z_dim
		self.n = n_dim
		self.curv = kwargs.get("c", 1.0)
		self.activ = kwargs.get("activation", nn.LeakyReLU())
		self.dropout = kwargs.get("dropout", 0.2)
		self.mannifold = geoopt.manifolds.PoincareBall(c=self.curv)
		self.learn_param = nn.Linear(n_dim, z_dim)
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		super(HAE, self).__init__()
	
	def encoder(self, inputs):
		enc = nn.Sequential([
		nn.Linear(self.x, self.n),
		self.activ,
		nn.Dropout(p=self.dropout)
		nn.Linear(self.n, self.n),
		self.activ,
		nn.Dropout(p=self.dropout)])
		e = enc(inputs)
		param = self.learn_param(e)
		mu = self.manifold.expmap0(param)
		log_var = nn.softplus(mu).add(1e-5)
		return mu, log_var

	def reparameterize(self, mu, log_var, K):
		dist = wrapped_hyper_norm(mu, log_var, self.curv)
		resample = dist.rsample(torch.([K]).to(device)
		return resample

	def decoder(self, embeddings, Geo = True):
		decode = nn.Sequential([
		nn.Linear(self.z, self.n)
		nn.activ,
		nn.Dropout(p=self.dropout),
		nn.Linear(self.n, self.n),
		nn.activ,
		nn.Dropout(p=self.dropout)])
		if Geo:
			dec = GeodesicLayer(embeddings)
		else:
			dec = self.manifold.logmap0(embeddings)
		recon = dec(emb)
		return recon 
	
	def forward(self, inputs, K=1):
		mu, log_var = self.encoder(inputs)
		embeddings = self.reparameterize(mu, log_var, K)
		z = self.decoder(embeddings)
		return z, mu, log_var

	def to_latent(self, inputs):
		with torch.no_grad():
			latent_emb = self.forward(inputs)
		return latent_emb.cpu().numpy()

	def plot_2D_embeddings(self, inputs, title='PVAE Latent Space',fig_out='./latent_emb.pdf', labels=None):
		latent_emb = self.to_latent(inputs)
		latent_emb = pd.DataFrame(latent_emb, names =("dim1", "dim2")]
		colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B',
                  '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                  '#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B',
                  '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
                  '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
                  '#CCEBC5', '#FFED6F']
		
		fig = plt.figure(19, 18)
		ax = plt.gca()
		circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
		ax.add_patch(circle)
		ax.plot(0, 0, 'x', c=(0, 0, 0), ms=4)
		if not (labels is None):
			latent_emb['labels'] = labels
			for i in set(labels):
				subset = latent_emb[latent_emb['labels'] == i]
				plt.scatter(subset['dim1'], subset'dim2'], label = i)  
				ax.legend(fontsize=fs, loc='outside', bbox_to_anchor=bbox)
		else:
			plt.scatter(latent_emb['dim1'], latent_emb['dim2'])
		plt.title(title)
		plt.xlabel('PVAE_1')
		plt.ylabel('PVAE_2')
		plt.show()
		plt.savefig(fig_out, dpi = 600)
	

def loss_func(model, inputs, dist):
	recon, mu, log_var = model.forward(inputs)
	
	mle_posterior = nn.MSELoss(recon_loss
