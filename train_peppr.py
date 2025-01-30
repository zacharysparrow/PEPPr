import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import csv
import pickle

ann_id_num = 'single_model'
if len(sys.argv) == 2:
    ann_id_num = str(sys.argv[1])

### Some hyperparameters ###
n_features = 15
nlayers = 15 
nnwidth = 50 

n_epochs = 1000
train_rate = 0.001
###

### Load data ###
property_names = ["Low shear visc","High shear visc","Toughness","Stress at break","Strain at break"]
featureids = ['x'+str(n+1) for n in range(n_features)]
n_props = len(property_names)
# Regression data
train_datafile = "data/training_set.csv"
train_file = pd.read_csv(train_datafile)
trainx_file = train_file[featureids]
trainy_file = train_file[property_names]
x_train = trainx_file.values
y_train = trainy_file.values

# Feature scaling
sc_y = StandardScaler()
x = x_train 
y = sc_y.fit_transform(y_train)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Import test dataset seperately
test_datafile = "data/test_set.csv"
test_file = pd.read_csv(test_datafile)
testid = test_file["ID"]
testx_file = test_file[featureids]
testy_file = test_file[property_names]
x_test_data = testx_file.values
y_test_data = testy_file.values
x_test = torch.tensor(x_test_data, dtype=torch.float32)
y_test = torch.tensor(y_test_data, dtype=torch.float32)
###

### Define ANN
layers = [nn.Linear(n_features, nnwidth), nn.ELU()]
for _ in range(15):
    layers.extend([nn.Linear(nnwidth, nnwidth), nn.ELU()])
layers.append(nn.Linear(nnwidth, n_props))
model = nn.Sequential(*layers)

mae_loss = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=train_rate, weight_decay=0.0)
###

### Optimize! ###
loss_log = []
loss_log_test = []
best_loss = 100
for step in range(n_epochs):
#    xx = x 
    pre = model(x)
    mae = mae_loss(pre, y)
    cost = mae 

#   see what's happening to the test error as a function of epoch
    pre_test = sc_y.inverse_transform(model(x_test).detach().numpy())
    pre_test = torch.tensor(pre_test, dtype=torch.float32)
    pre_train = sc_y.inverse_transform(pre.detach().numpy())
    pre_train = torch.tensor(pre_train, dtype=torch.float32)
    mae_test = mae_loss(pre_test, y_test)
    mae_train = mae_loss(pre_train, y_train)
    loss_log_test.append([mae_test.item()])
    loss_log.append([0.0, mae_train.item(), 0.0, cost.item()]) 
    
    if mae.item() < best_loss:
        best_loss = mae.item()
        torch.save(model, '/home/sparrow/Documents/Research/poly_regression/140blend/140Blend/pytorch/models/ann_lowvisc_'+ann_id_num+'.pt')
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print('- MAE train: %2.2f, MAE test: %2.2f' % (mae_train.item(), mae_test.item()))
###

### Save the model ###
torch.save(model, 'models/peppr_'+ann_id_num+'.pt')

# Save the scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(sc_y, f)

# writing to csv file
np.savetxt('models/loss_log_'+ann_id_num+'.csv', np.array(loss_log), delimiter=" ", fmt='%f')
np.savetxt('models/loss_log_test_'+ann_id_num+'.csv', np.array(loss_log_test), delimiter=" ", fmt='%f')
