import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import geoopt
from geoopt.optim import RiemannianAdam, RiemannianSGD
import matplotlib.pyplot as plt 
from VAE import HVAE
from Modules import *
from WrappedNormal import WrappedNormal
from Loss import hyp_loss 
from Data import get_dataloaders
import timeit
#Training functions for the VAE model
def mse_loss(inputs, targets):
    return torch.sum((inputs - targets) ** 2)


def train(trainloader, model, obj, optimizer, epoch, device):
    model.train()
    train_loss = 0
    for i, data in enumerate(trainloader):
        data = data.to(device)   
        optimizer.zero_grad()
        loss = obj(model, data)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    train_loss /= len(trainloader)
    print(f"{epoch}: train_loss = {np.mean(train_loss):.3e}")
    return train_loss

def test(testloader, model, obj, optimizer, epoch, device):
    model.eval()
    test_loss = 0
    MSE = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            data = data.to(device)
            loss = obj(model, data)
            mse =  mse_loss(model.reconstruct(data), data)
            test_loss += loss.item()
            MSE += mse.item()
        
    test_loss /= len(testloader)
    MSE /= len(testloader)
    print(f"{epoch}: test_loss = {np.mean(test_loss):.3e} MSE = {np.mean(MSE):.3e}")
    return test_loss

def train_model(trainloader, testloader, model, obj, optim, epochs, cuda, gui, model_output, validataion_output):
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("training on GPU") if str(device) == "cuda:0" else print("training on CPU")
    else:
        device = torch.device("cpu")
    
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        t_start = timeit.default_timer()
        x = train(trainloader, model, obj, optim, epoch, device)
        y = test(testloader, model, obj, optim, epoch, device)
        elapsed = timeit.default_timer() - t_start
        train_losses.append(x)
        test_losses.append(y)
        print(f"{epoch}: train_time={elapsed:.3f}s")
    plt.plot(range(epochs), train_losses, label='trian')
    plt.plot(range(epochs), test_losses, label='test')
    plt.title("Validation")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig(validataion_output, dpi=600)
    if gui:
        plt.show()
    torch.save(model.state_dict(), model_output)
    return model 

# Debug Code
#  Comment Out
if __name__ == '__main__':
    trainloader, testloader = get_dataloaders(2, 32, "../PoincareEmbeddingsForEpigentics/code/datasets/TCDD_data.csv", True, 0.33, True)
    x = 5889
    z = 2
    n = 2
    n_size = 256
    activ = nn.LeakyReLU()
    drop_rate = 0.2
    manifold = PoincareBall(c=1.0)
    encoder = WrappedEncoder(x, z, n, n_size, activ, drop_rate, manifold)
    decoder = WrappedDecoder(z, x, n, n_size, activ, drop_rate, manifold)
    prior = WrappedNormal
    posterior = WrappedNormal
    likelihood = Normal
    
    model = HVAE(encoder, decoder, 
    prior, posterior, likelihood).double()
    optim = RiemannianAdam(model.parameters(), 1e-3)
    train_model(trainloader, testloader, model, hyp_loss, optim, 20, True, True, "./model.pth", "./validation_graph.pdf")
    
