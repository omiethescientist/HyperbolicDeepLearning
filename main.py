import sys
import torch
import torch.nn as nn
import geoopt
from geoopt.manifolds import PoincareBall
from geoopt.optim import RiemannianSGD, RiemannianAdam
import os
import pandas as pd
import argparse
from VAE import HVAE
from Modules import *
from  Loss import hyp_loss
from Data import get_dataloaders
from Train import train_model
from WrappedNormal import WrappedNormal
from torch.distributions import Normal

#Parse Terminal Arguements
parser = argparse.ArgumentParser(description='Run Poincare Variatonal Autoencoder for RNA-Seq Data Sets')

parser.add_argument('inputs', metavar='i', type=str, help='path to RNA-Seq Data to process')
parser.add_argument('--input-dim', metavar='x', type=int, help='Dimensions or number of genes being used in dataset')
parser.add_argument('--latent-dim', metavar='z', type=int, help='Dimensions of latent space in VAE')
parser.add_argument('--output-model', metavar='om', type=str, help='path to output pytorch model')
parser.add_argument('--output-valid', metavar='ov', type=str, help='path to output validation plot')
parser.add_argument('--labels', metavar='lbs', type=int, help='Is the last column a label column?')
#Customizations
parser.add_argument('--hidden-lay', metavar='n', type=int, default = 2, help='Number of hidden layers for encoder and decoder')
parser.add_argument('--hidden-size', metavar='ns', type=int, default = 256, help='Size of hidden layers for encoder and decoder')
#parser.add_argument('--activation', metavar='nl', type=str, default = 'LeakyReLU', help='Non-linear activation function for neural net')
parser.add_argument('--drop-rate', metavar='dr', type=float, default = 0.2, help='Drop rate for nn.Dropout layer')
parser.add_argument('--curvature', metavar='c', type=float, default = 1.0, help='Curvature of poincare ball manifold')

#parser.add_argument('--encoder', metavar='e', type=str, default = 'MobiusEncoder', help='Encoder Architechture', choices=['WrappedEncoder', 'MobiusEncoder'])
#parser.add_argument('decoder', metavar='d', type=str, default = 'MobiusDecoder', help='Decoder Architechture', choices=['WrappedDecoder', 'MobiusDecoder', 'GeodesicDecoder'])

#Probability Distributions
#parser.add_argument('--prior', metavar='pz', type=str, default = 'WrappedNormal', help='Prior for encoder and data generator', choices=['WrappedNormal', 'Normal'])
#parser.add_argument('--posterior', metavar='qzx', type=str, default = 'WrappedNormal', help='Posterior for encoder and generator', choices=['WrappedNoraml', 'Normal']) 
#parser.add_argument('--likelihood', metavar='pxz', type=str, default = 'Normal', help='Likelihood distribution for decoder and reconstruction', choices = ['Normal', 'Bernoulli'])

#Optimization
#parser.add_argument('--optim', metavar='o', type=str, default  =  'RiemannianAdam', help='Optimizer for the Neural Network', choices = ['RiemannianAdam', 'RiemannianSGD'])
parser.add_argument('--lr', type=float, default = 1e-3, help='Learning rate for the optimizer')
parser.add_argument('--epochs', metavar='e', type=int, default = 250, help = 'Number of epochs for training model')

#Data Loaders
parser.add_argument('--batch-size', metavar='bs', type=int, default = 64, help='Number of samples per batch into model')
parser.add_argument('--workers', metavar='w', type=int, default = 1, help='Number of workers for dataloader')
parser.add_argument('--valid-split', type = float, default = 0.33, help='test proportion of data train test split')

#For Numerical Stability
parser.add_argument('--double', metavar='d', type=int, default = 1, help='For purposes of numerical stability the modeluses double precision floats due as per the advice of the geoopt documentation. However, these models are memory intensive.')

#Cuda or CPU
parser.add_argument('--cuda', type=int, default = 0, help='Use Cuda for GPU boost.')

args = parser.parse_args()

inputs = args.inputs
workers = args.workers
batch_size = args.batch_size

if args.cuda == 1:
	cuda = True
else:
	cuda = False

if args.labels == 1:
	labels = True
else:
	labels = False

if args.double == 1:
	double = True
else:
	double = False

trainloader, testloader = get_dataloaders(workers, batch_size, inputs, labels, 0.33, double)
x = args.input_dim
z = args.latent_dim
n = args.hidden_lay
n_size = args.hidden_size
activ = nn.LeakyReLU()
drop_rate = args.drop_rate
manifold = PoincareBall(c=args.curvature)
encoder = WrappedEncoder(x, z, n, n_size, activ, drop_rate, manifold)
decoder = MobiusDecoder(z, x, n, n_size, activ, drop_rate, manifold)
prior = WrappedNormal
posterior = WrappedNormal
likelihood = Normal

if double:
	model = HVAE(encoder, decoder, prior, posterior, likelihood).double()
else:

	model = HVAE(encoder, decoder, prior, posterior, likelihood).double()
optim = RiemannianSGD(model.parameters(), args.lr)
HVAE = train_model(trainloader, testloader, model, hyp_loss, optim, args.epochs, cuda, False, args.output_model, args.output_valid)
if z == 2:
	if labels:
		data = pd.read_csv(args.inputs, sep=',')
		x_in = data.iloc[:, :-1].values
		labels = data.iloc[:, -1]
	else:
		x_in = pd.read_csv(args.inputs, sep=',').values
		labels = None
	
	x_in = torch.Tensor(x_in)
	if double:
		x_in = x_in.double()
	HVAE.plot_2D_embeddings(x_in, labels = labels)

