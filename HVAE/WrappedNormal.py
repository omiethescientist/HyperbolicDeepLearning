import torch
import torch.nn as nn
from geoopt.manifolds import PoincareBall
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from geoopt.manifolds.poincare.math import *
from torch.distributions.utils import _standard_normal
from torch.distributions import Normal

# Code from https://github.com/emilemathieu/pvae/blob/master/pvae/distributions/wrapped_normal.py
# Code by Emile Mathieu from Microsoft Research and Oxford Stats
# Paper: https://arxiv.org/pdf/1901.06033.pdf
# Define Wrapped Normal Distribution on pyTorch infrastructure

class WrappedNormal(Distribution):
	
	arg_constraints = {'mu': constraints.real, 'Sigma': constraints.positive}
	support = constraints.real
	has_rsample = True
	
	@property
	def mean(self):
		return self.mu
	
	@property
	def stddev(self):
		return NotImplementedError 
	
	@property
	def scale(self):
		return self.Sigma

	def __init__(self, mu, Sigma, c = 1, dim = 2):
		self.c = torch.Tensor([c])
		self.mu = mu
		self.Sigma = Sigma
		self.manifold = PoincareBall(c)
		batch_shape = self.mu.shape[:-1]
		event_shape = torch.tensor([dim])
		self.zeros = torch.zeros(1, event_shape).to(mu.device)
		super(WrappedNormal, self).__init__(batch_shape, event_shape)
	
	def sample(self, sample_size =torch.Size()):
		with torch.no_grad():
			return self.rsample(sample_size)
	
	#Sampling function for reparameterization
	def rsample(self, sample_size = torch.Size()):
		v = self.Sigma.mul(_standard_normal(sample_size, dtype=self.mu.dtype, device=self.mu.device))
		v = v.div(self.manifold.lambda_x(self.zeros, keepdim = True))
		u = self.manifold.transp(self.zeros, self.mu, v) 			
		z = self.manifold.expmap(self.mu, u)
		return z
	
	def log_prob(self, z):
		#Sample from N(\lambda^c_\mu * log_\mu(z)|0, \Sigma)
		log_mu_z = self.manifold.logmap(self.mu, z)
		log_mu_z = self.manifold.transp(self.mu, self.zeros, log_mu_z)
		u = log_mu_z.mul(self.manifold.lambda_x(self.zeros, keepdim = True))
		N_pdf = Normal(torch.zeros_like(self.Sigma), self.Sigma).log_prob(u).sum(-1,keepdim=True)
		
		#Calculate the Wrapped Change of Variable for the Poincare Ball
		sqrtC = torch.sqrt(self.c).to(self.mu.device)
		d = self.manifold.dist(self.mu, u, keepdim=True)		
		logdetexp = (self._event_shape.to(self.mu.device).sub(1)).mul(torch.sinh(sqrtC.mul(d)).div(sqrtC).div(d)).log()
		return N_pdf - logdetexp
#Debug Code
#Comment Out	
#if __name__ == '__main__':
#	mu = torch.rand(64,2)
#	sigma = torch.rand(64,2)
#	z = torch.rand(64, 2)
#	distr = WrappedNormal(mu, sigma)
#	sampl = distr.sample([64,2])
#	logProb = distr.log_prob(z)
#	print(sampl)
#	print(logProb)

