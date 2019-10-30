import numpy as np

# network parameters
# generator first layer 'lstm' or 'dense'
layer = 'lstm'
# critic structure 'dense' or 'convolution'
C_layer = 'convolution'
# generator layer structure 'l' for only lstm 'ld' for lstm with dense layers
G_layer = 'l'
# date for train-test split
date = '2016-01-04'
# data name
data_name = 'Merge_ORCL.csv'
# forecast horizon (0 if i forecast a timeseries)
# timesteps per input
n_steps = 24
# lstm size generator
lstm_G = np.array([256, 128])
# lstm size critic (does not work at the moment)
lstm_C = np.array([256, 128])
# dense size Generator
dense_G = np.array([128, 128, 128, 64])
# dense size critic (Only dense layers in critic)
dense_C = np.array([128, 128, 64, 32])
# dense size critic (for convolutional critic)
dense_C_C = np.array([256, 128])
# convolutional layers
conv_C = np.array([32, 64, 128])
# use reservoir or not
reservoir = False
# ouputstyle 'ts' or 'mean_sd'
outputstyle = 'ts'
# ouputstyle_feature 'ts', 'mean', 'std', 'var' or 'var_t' (the forecasted feature we test our feature importance for)
outputstyle_feature = 'std'
# output length generator
ts_out_length = 5
# how many batches to prefetch in Pipeline
prefetch_batch = 1
# how many times to repeat the dataset for one epoch
repeat = 3
# samples for VaR calculation
samples = 3
# alpha VaR
alpha = .05
# all alphas
alpha_all = [.01, .025, .05, .1, .9, .95, .975, .99]
# Batch Size
batch_size = 128
# rolling window size for Var mean and std
rolling_var = 15
# noise size
noise_size = 20
# degrees of freedom for t-distr
dof = 5
# Optimizer
optimizer = 'radam'

# Reservoir parameters:
hor = 0
# dimension of Input
n = 19
# number of Neurons
n_neu = 10
# number of time series values
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
n_input_layers = 20

