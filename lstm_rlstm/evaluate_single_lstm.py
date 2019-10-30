import numpy as np
import matplotlib.pyplot as plt
import parameters as par
import functions as fu
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
from scipy.stats import t
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

load_scaler = True

# Data
mean_std = pd.read_csv(par.data_name, delimiter=',')[['mean', 'std']].to_numpy()
data = pd.read_csv(par.data_name, delimiter=',')[['volume', 'RVOL', 'MACD', '20sd', 'High', 'Low', 'Adj Close',
                                                  'dt_diff', 'r_fft_100', 'r_fft_500', 'logreturns']]
data_y = pd.read_csv(par.data_name, delimiter=',')[['logreturns']]
dates = pd.read_csv(par.data_name, delimiter=',')[['dt']]
n_split = fu.get_datesplit(par.date, dates)
if par.reservoir:
    data = np.load('reservoir.npz')
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    if par.outputstyle in ['mean_sd', 'var', 'var_t']:
        returns_var_test = data['returns_var_test.npy']
    features = train_x.shape[2]
else:
    if par.outputstyle in ['mean_sd', 'var', 'var_t']:
        train_x, test_x, train_y, test_y, features, returns_var_test = \
            fu.get_data_raw(data, data_y, par.outputstyle, n_split, par.rolling_var, par.n_steps,
                            par.ts_out_length, par.hor, par.alpha, par.layer, par.reservoir, par.dof, load_scaler)
    elif par.outputstyle == 'ts':
        train_x, test_x, train_y, test_y, features = \
            fu.get_data_raw(data, data_y, par.outputstyle, n_split, par.rolling_var, par.n_steps,
                            par.ts_out_length, par.hor, par.alpha, par.layer, par.reservoir, par.dof, load_scaler)


# Load Model
model = keras.models.load_model(par.model_name + '.h5')
model.load_weights(par.model_name + '.weights.h5')
# Model Summary
model.summary()

pred = model.predict(test_x)
predt = model.predict(train_x)

if par.outputstyle in ['var', 'var_t']:
    noe = np.sum(returns_var_test[:, 0] - test_y[:, 0] < 0)
    noe_nn = np.sum(returns_var_test[:, 0] - pred[:, 0] < 0)
    print(noe/test_y.shape[0])
    print(noe_nn/test_y.shape[0])
    plt.plot(np.arange(0, pred.shape[0]), returns_var_test[:, 0], linestyle='solid', color='green', label='logreturns')
    plt.plot(np.arange(0, pred.shape[0]), test_y[:, 0], linestyle='solid', color='blue', label=par.outputstyle)
    plt.plot(np.arange(0, pred.shape[0]), pred[:, 0], linestyle='solid', color='red', label='prediction')
    plt.legend(loc='upper right')
    plt.show()
    if par.outputstyle == 'var':
        pd.DataFrame.from_records(
            {'logreturns': returns_var_test[:, 0], 'var_n': pred[:, 0]}).to_csv(
            'cond_cov_n.csv', sep=',', index=False)
    else:
        pd.DataFrame.from_records(
            {'logreturns': returns_var_test[:, 0], 'var_t': pred[:, 0]}).to_csv(
            'cond_cov_t.csv', sep=',', index=False)
elif par.outputstyle == 'mean_sd':
    VaR_list = []
    VaR_t_list = []
    VaR_real = (test_y[:, 0] + test_y[:, 1] * np.sqrt((par.rolling_var + 1) / par.rolling_var) * t.ppf(
        par.alpha, par.rolling_var))
    VaR_t_real = (test_y[:, 0] + test_y[:, 1] * np.sqrt((par.dof - 2) / par.dof) * t.ppf(par.alpha, par.dof))
    noe_n = np.sum(returns_var_test[:, 0] - VaR_real < 0)
    noe_t = np.sum(returns_var_test[:, 0] - VaR_t_real < 0)
    print(noe_n / returns_var_test.shape[0])
    print(noe_t / returns_var_test.shape[0])
    plt.plot(np.arange(0, returns_var_test.shape[0]), returns_var_test[:, 0], linestyle='solid', color='red',
             label='logreturns')
    plt.plot(np.arange(0, returns_var_test.shape[0]), VaR_real, linestyle='solid', label='VaR_t')
    plt.title('VaR_real')
    plt.savefig('var_real_' + str(par.alpha) + '_' + str(par.rolling_var) + '.png')
    plt.show()
    plt.plot(np.arange(0, returns_var_test.shape[0]), returns_var_test[:, 0], linestyle='solid', color='red',
             label='logreturns')
    plt.plot(np.arange(0, returns_var_test.shape[0]), VaR_t_real, linestyle='solid', label='VaR_t')
    plt.title('VaR_t_real')
    plt.savefig('var_t_real_' + str(par.alpha) + '_' + str(par.rolling_var) + '.png')
    plt.show()
    for i in range(len(par.alpha_all)):
        VaR = (pred[:, 0] + np.abs(pred[:, 1]) * np.sqrt((par.rolling_var + 1) / par.rolling_var) * t.ppf(
            par.alpha_all[i], par.rolling_var))
        VaR_t = (pred[:, 0] + np.abs(pred[:, 1]) * np.sqrt((par.dof - 2) / par.dof) * t.ppf(par.alpha_all[i], par.dof))
        noe_n = np.sum(returns_var_test[:, 0] - VaR < 0)
        noe_t = np.sum(returns_var_test[:, 0] - VaR_t < 0)
        print(noe_n / returns_var_test.shape[0])
        print(noe_t / returns_var_test.shape[0])
        plt.plot(np.arange(0, returns_var_test.shape[0]), returns_var_test[:, 0], linestyle='solid', color='red', label='logreturns')
        plt.plot(np.arange(0, returns_var_test.shape[0]), VaR, linestyle='solid', label='VaR')
        plt.title('VaR')
        plt.savefig('var_n_' + str(par.alpha_all[i]) + '.png')
        plt.show()
        plt.plot(np.arange(0, returns_var_test.shape[0]), returns_var_test[:, 0], linestyle='solid', color='red', label='logreturns')
        plt.plot(np.arange(0, returns_var_test.shape[0]), VaR_t, linestyle='solid', label='VaR_t')
        plt.title('VaR_t')
        plt.savefig('var_t_' + str(par.alpha_all[i]) + '.png')
        plt.show()
        plt.plot(np.arange(0, returns_var_test.shape[0]), test_y[:, 0], linestyle='solid', color='blue', label='mean')
        plt.plot(np.arange(0, returns_var_test.shape[0]), pred[:, 0], linestyle='solid', color='red', label='prediction')
        plt.title('meanpred')
        plt.legend(loc='upper right')
        plt.show()
        plt.plot(np.arange(0, returns_var_test.shape[0]), test_y[:, 1], linestyle='solid', color='blue', label='sd')
        plt.plot(np.arange(0, returns_var_test.shape[0]), np.abs(pred[:, 1]), linestyle='solid', color='red', label='prediction')
        plt.title('sdpred')
        plt.legend(loc='upper right')
        plt.show()

        VaR_list.append(VaR)
        VaR_t_list.append(VaR_t)
    VaR_out = np.vstack(VaR_list).transpose()
    VaR_t_out = np.vstack(VaR_t_list).transpose()
    pd.DataFrame.from_records({'logreturns': returns_var_test[:, 0],
                               'var_n_01': VaR_out[:, 0], 'var_n_025': VaR_out[:, 1], 'var_n_05': VaR_out[:, 2],
                               'var_n_1': VaR_out[:, 3], 'var_n_9': VaR_out[:, 4], 'var_n_95': VaR_out[:, 5],
                               'var_n_975': VaR_out[:, 6], 'var_n_99': VaR_out[:, 7],
                               'var_t_01': VaR_t_out[:, 0], 'var_t_025': VaR_t_out[:, 1], 'var_t_05': VaR_t_out[:, 2],
                               'var_t_1': VaR_t_out[:, 3], 'var_t_9': VaR_t_out[:, 4], 'var_t_95': VaR_t_out[:, 5],
                               'var_t_975': VaR_t_out[:, 6], 'var_t_99': VaR_t_out[:, 7], }
                              ).to_csv('cond_cov.csv', sep=',', index=False)
elif par.outputstyle == 'ts':
    print('No plots to show')
else:
    print('choose outputstyle from "mean_sd", "var", "var_t", "ts" ')
