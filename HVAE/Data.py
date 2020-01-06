import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
#Get Training and Testing DataLoaders for pytorch models I make

def get_dataloaders(workers, batch_size, csv_path, labels, test_split, double):
    data = pd.read_csv(csv_path, sep=',')
    if labels:
        data = data.iloc[:, :-1]
    X_train, X_test = train_test_split(data, test_size = test_split, random_state = 42)
    X_test = torch.Tensor(X_test.values)
    X_train = torch.Tensor(X_train.values)
    if double:
        X_test = X_test.double()
        X_train = X_train.double()
    trainloader = DataLoader(X_train, batch_size = batch_size, shuffle = False, num_workers = workers)
    testloader = DataLoader(X_test, batch_size = batch_size, shuffle = False, num_workers = workers)
    return trainloader, testloader

# Debug Code
# Comment Out
#if __name__ == "__main__":
#    worker = 2
#    batch_size = 16
#    csv_path = "../PoincareEmbeddingsForEpigentics/code/datasets/TCDD_Data.csv"
#    labels = True
#    test_split = 0.33
#    print(get_dataloaders(worker, batch_size, csv_path, labels, test_split))
    
    
    

    
