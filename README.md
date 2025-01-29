# PEPPr
This repository contains supplementary code and data to accompany the manuscript entitled "Designing Polymers with Molar Mass Distribution Based Machine Learning" by Jenny Hu, Brian G. Ernst, Zachary M. Sparrow, Spencer M. Mattes, Geoffrey W. Coates, Robert A. DiStasio Jr., and Brett P. Fors. It is a copy of the [official repository](https://github.com/robdistasio/PEPPr), which is also written by me but hosted by Robert DiStasio (the lab PI) with slight modifications/additions.

The included data (training_set.csv, test_set.csv, inverse_set.csv, and valorization_set.csv), scripts (eval_peppr.py), and models (peppr_[1-100].pt) contain all of the information needed to reproduce the main results reported in this manuscript.

The machine learning model developed in this work is PEPPr (**P**oly**E**thylene **P**roperty **Pr**edictor) uses an ensemble of feed forward neural networks (and pytorch) to predict select rheological and mechanical properties of high density polyethylene (HDPE) based on the molecular weight distribution (MWD).

System requirements:

python 	3.11.4  
numpy   1.25.2  
pytorch 2.0.1  
sklearn 1.3.0  
pandas 	1.5.3

## Instructions

After installing the required packages and cloning this repository, the python script for evaluating the PEPPr ensemble on a given data set can be run with the following command:
python eval_peppr.py /data/path.csv
where /data/path.csv points to the .csv file that contains the features for the data that you would like to use to evaluate the PEPPr model. 

For example, inference on the test set using PEPPr can be performed via:
python eval_peppr.py /data/test_set.csv

The output will be a .csv file containing the means and standard deviations of the PEPPr ensemble predictions corresponding to the data in the original .csv file. 
Predictions are the logarithmically transformed properties with the following units:

Low shear viscosity: Pa·s  
High shear viscosity: Pa·s  
Toughness: MJ·m^(-3)·10^(-2)  
Stress at break: MPa  
Strain at break: %

Standard deviations are computed in the logarithmically-transformed space and can be used to gauge model uncertainty for each prediction.

IDs for the inverse and valorization set are in the form:  
X-X-X-X-X-X-X-X-X-X  
where each X corresponds to the fraction of the following base polymers used to "computationally blend" the data point:  
PE38 V3 - PE98 V2 - PE121 - PE198 - PE277 - PE8 V2 - PE18 - PE357 V3 - commercial degraded - post consumer degraded  
Note that the commercial degraded and post comsumer degraded entries are not applicable for the inverse set, resulting in IDs that contain two fewer entries.

Inference on all of the data sets included herein should run in a matter of seconds on a standard desktop or laptop computer.

We note that all ML models are stored in models/ with the .pt file extension, so no training is required to use PEPPr.
Please see eval_peppr.py for a demonstration of how to load the models and use them for inference.

## Reference
If you use this repository, please cite the following work:
**COMING SOON**

## TODO
- upload training script
- add reference
