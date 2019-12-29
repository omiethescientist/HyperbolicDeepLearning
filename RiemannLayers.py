import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import geoopt
from geoopt.manifolds.poincare.math import arsinh, mobius_add
import math
from geoopt.manifolds import PoincareBall
from geoopt import ManifoldParameter

# Code from https://github.com/emilemathieu/pvae/blob/master/pvae/ops/manifold_layers.py
# Code by Emile Mathieu from Microsoft Research and Oxford Stats
# Paper: https://arxiv.org/pdf/1901.06033.pdf
# Define Base Riemann Layer and Geodesic Layer

class RiemannLayer(nn.Module):
	
	def __init__(self, in_features, out_features, manifold, over_param):
		super(RiemannLayer, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(out_features, in_features))
		self.over_param = over_param
		if self.over_param:
			self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
		else:
			self._bias = Parameter(torch.Tensor(out_features, 1))
		self.manifold = manifold
		self.reset_parameters()
	
	@property
	def weight(self):
		return self.manifold.transp0(self._bias, self._weight)
	
	@property
	def bias(self):
		return self.manifold.expmap0(self.weight.mul(self._bias))
    
	def reset_parameters(self):
		nn.init.kaiming_normal_(self.weight, a =math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 4 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)
			if self.over_param:
				with torch.no_grad(): self._bias.set_(self.manifold.expmap0(self._bias))

class GeodesicLayer(RiemannLayer):
	def __init__(self, in_features, out_features, manifold):
		super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param=False)
		self.c = self.manifold.c
	def forward(self, inputs):
		sqrtC = math.sqrt(self.c)
		if len(inputs.shape) < 3:
			x = inputs.unsqueeze(-2).expand(inputs.shape[0], self.out_features, self.in_features)
		else:
			x = inputs.unsqueeze(-2).expand(*inputs.shape[:-(len(inputs.shape)-2)], self.out_features, self.in_features)
		a = self.bias
		p = self.weight
		mob_diff = self.manifold.mobius_add(-p, x)
		mob_diff_sqr = mob_diff.pow(2).sum(dim=-1,keepdim=False).clamp(1e-15)
		sc_mob_diff_a = mob_diff.mul(a).sum(dim=-1,keepdim=False)
		a_norm = a.norm(keepdim=False, p=2).clamp_min(1e-15)
		numerator = 2 * sqrtC * sc_mob_diff_a
		denomenator = (1-self.c*mob_diff_sqr) * a_norm
		res = arsinh(numerator / denomenator.clamp(1e-15) / sqrtC)	
		res = res * a_norm
		return res

class MobiusLayer(RiemannLayer):
    def __init__(self, in_features, out_features, manifold):
        super(MobiusLayer, self).__init__(in_features, out_features, manifold, over_param=False)
        
    def forward(self, inputs):
        res = self.manifold.mobius_matvec(self.weight, inputs)
        return res

#Debugging Code
#Comment Out
#if __name__ == '__main__':
#	in_features = 2
#	out_features = 256
#	c = 1
#	dim = 2
#	manifold = PoincareBall(c=1)
#	Layer = GeodesicLayer(in_features, out_features, manifold)
#	MobLay = MobiusLayer(in_features, out_features, manifold)
#	inputs = torch.randn(64,2)
#	print(Layer(inputs))
#	print(MobLay(inputs))
#	print(MobLay(inputs).shape)