#!/home/sparrow/anaconda3/envs/pytorch_env/bin/python

import sys
from pathlib import Path
import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd

print("Input: "+sys.argv[1])

n_features = 15
n_models = 100
with open("ensemble_size.txt", "r") as file:
    n_models = int(file.readline())
featureids = ['x'+str(n+1) for n in range(n_features)]
properties = ["Low shear visc.","High shear visc.","Toughness","Stress at break","Strain at break"]

### Load data ###
with open('models/scaler.pkl','rb') as f:
    sc_y = pickle.load(f)
test_datafile = sys.argv[1] 
test_file = pd.read_csv(test_datafile)
testid = test_file["ID"]
testx_file = test_file[featureids]
x_test_data = testx_file.values
x_test = torch.tensor(x_test_data, dtype=torch.float32)
###

train_predictions = []
test_predictions = []
for i in range(n_models):
    ### Load and run the model ###
    model = torch.load('models/peppr_'+str(i+1)+'.pt')
    model.eval()
    models_result = np.array(model(x_test).data.numpy())
    models_result = sc_y.inverse_transform(models_result)
    test_predictions.append(models_result.tolist())

test_predictions = np.array(test_predictions)

# Make the ensemble prediction
mean_values_test = np.mean(test_predictions, axis=0)
stdev_values_test = np.std(test_predictions, axis=0)
peppr_output = np.concatenate((mean_values_test,stdev_values_test), axis=1)

csv_header = properties + ["St.Dev. " + item for item in properties]
file_root=Path(sys.argv[1]).stem
dataframe_to_write = pd.DataFrame(peppr_output)
dataframe_to_write.index = testid
dataframe_to_write.to_csv(file_root+'_predictions.csv', sep=',', header=csv_header)
print("Output: "+file_root+"_predictions.csv")





