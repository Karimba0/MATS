import numpy as np

# Network parameters:
# Generator first layer 'lstm' or 'dense'
layer = 'lstm'
# Model name
model_name = 'normal_lstm_radam'
# Model name rolling
model_name_rolling = 'rolling_lstm_radam'
# Specifiy the stock used
data_name = 'Merge_LOW.csv'
# Date for train-test split
date = '2016-01-04'
# timesteps per input
n_steps = 15
# lstm neurons
lstm = np.array([64, 0])
# dense neurons Generator
dense = np.array([128, 128, 32])
# Use reservoir or not
reservoir = False
# rolling or single LSTM produced in init_LSTM.py
rolling = False
# ouputstyle 'ts', 'mean_sd', 'var' or 'var_t' (what we want to fit the model to)
outputstyle = 'mean_sd'
# outputlength if 'ts' is selected (ts means we fit to future logreturns)
ts_out_length = 5
# alpha VaR
alpha = .05
# all tested alphas
alpha_all = [.01, .025, .05, .1, .9, .95, .975, .99]
# rolling window size for Var
rolling_var = 15
# degrees of freedom for t-distr
dof = 5

# Reservoir parameters:
# Forecast horizon
hor = 5
# Dimension of Input
n = 19
# Number of Neurons
n_neu = 10
# Number of time series values
n_i = 1
# has to be int
k_i = n_neu//n_i
# separation of Neurons
theta = 1.27
# delay
tau = theta*n_neu
# h
h = np.log(1+theta)
# number of reservoirs
n_res = 15
# number of neuron layers in one input
n_input_layers = 30

