import numpy as np
import matplotlib.pyplot as plt
import parameters as par
import functions as fu
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
from scipy.stats import t
from keras_radam.training import RAdamOptimizer

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


num_epochs = 1000
batch_size = 128
tr = 2000
te = 110
noe_sum = 0
noe_nn_sum = 0
# learning rate for continue = False set to 1e-10 because model gets trained continuously on all subsets
# learning rate for continue = True
lr = 1e-3
# True if training pretrained rolling model (set to false if model was only produced by init_lstm.py)
_continue = True
# True if we only want to evaluate the rolling model
evaluate = True
load_scaler = True


# Data
data = pd.read_csv(par.data_name, delimiter=',')[['volume', 'RVOL', 'MACD', '20sd', 'High', 'Low', 'Adj Close',
                                                  'dt_diff', 'r_fft_100', 'r_fft_500', 'logreturns']]
data_y = pd.read_csv(par.data_name, delimiter=',')[['logreturns']]
dates = pd.read_csv(par.data_name, delimiter=',')[['dt']]
# split date minus train size to get first forecast on split date
n_split = fu.get_datesplit(par.date, dates) - tr
# logreturns for noe calculation
returns_var = data[['logreturns']].to_numpy()[n_split + tr:]
if par.reservoir:
    data = np.load('reservoir.npz')
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_x = np.vstack((train_x[-tr:, :, :], test_x))
    test_y = data['test_y']
    test_y = np.vstack((train_y[-tr:, :], test_y))
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

# Prediction Lists
pred_list = []
predt_list = []
te_y_list = []
# Number of networks
steps = (test_x.shape[0] - tr)//te
model_name = par.model_name_rolling + '_' + str(te) + '_' + str(tr) + '_'
if evaluate:
    for i in range(steps):
        tr_x = test_x[i * te:i * te + tr, :, :]
        tr_y = test_y[i * te:i * te + tr, :]
        te_x = test_x[i * te + tr:(i + 1) * te + tr, :, :]
        te_y = test_y[i * te + tr:(i + 1) * te + tr, :]
        model = keras.models.load_model(model_name + str(i) + '.h5')
        model.load_weights(model_name + str(i) + '.weights.h5')
        model.summary()
        pred = model.predict(te_x)
        predt = model.predict(tr_x)
        pred_list.append(pred)
        te_y_list.append(te_y)
    pred_out = np.vstack(pred_list)
    te_y_out = np.vstack(te_y_list)

else:
    if _continue:
        for i in range(steps):
            tr_x = test_x[i * te:i * te + tr, :, :]
            tr_y = test_y[i * te:i * te + tr, :]
            te_x = test_x[i * te + tr:(i + 1) * te + tr, :, :]
            te_y = test_y[i * te + tr:(i + 1) * te + tr, :]
            model = keras.models.load_model(model_name + str(i) + '.h5')
            model.load_weights(model_name + str(i) + '.weights.h5')
            model.summary()
            radam = RAdamOptimizer(learning_rate=lr)
            model.compile(loss='mae', optimizer=radam, metrics=['mae'])
            model.fit(tr_x, tr_y, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_data=(te_x, te_y))
            model.save(model_name + str(i) + '.h5')
            model.save_weights(model_name + str(i) + '.weights.h5')
            print('Saved model to disk')
            pred = model.predict(te_x)
            predt = model.predict(tr_x)
            pred_list.append(pred)
            te_y_list.append(te_y)
        pred_out = np.vstack(pred_list)
        te_y_out = np.vstack(te_y_list)
    else:
        model = keras.models.load_model(par.model_name_rolling + '.h5')
        model.load_weights(par.model_name_rolling + '.weights.h5')
        for i in range(steps):
            tr_x = test_x[i * te:i * te + tr, :, :]
            tr_y = test_y[i * te:i * te + tr, :]
            te_x = test_x[i * te + tr:(i + 1) * te + tr, :, :]
            te_y = test_y[i * te + tr:(i + 1) * te + tr, :]
            model.summary()
            radam = RAdamOptimizer(learning_rate=1e-10)
            model.compile(loss='mae', optimizer=radam, metrics=['mae'])
            model.fit(tr_x, tr_y, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_data=(te_x, te_y))
            model.save(model_name + str(i) + '.h5')
            model.save_weights(model_name + str(i) + '.weights.h5')
            print('Saved model to disk')
            pred = model.predict(te_x)
            predt = model.predict(tr_x)
            pred_list.append(pred)
            te_y_list.append(te_y)
        pred_out = np.vstack(pred_list)
        te_y_out = np.vstack(te_y_list)


if par.outputstyle in ['var', 'var_t']:
    noe = np.sum(returns_var[:, 0] - te_y_out[:, 0] < 0)
    noe_nn = np.sum(returns_var[:, 0] - pred_out[:, 0] < 0)
    print(noe / te_y_out.shape[0])
    print(noe_nn / te_y_out.shape[0])
    plt.plot(np.arange(0, pred_out.shape[0]), returns_var[:, 0], linestyle='solid', color='green', label='logreturns')
    plt.plot(np.arange(0, pred_out.shape[0]), te_y_out[:, 0], linestyle='solid', color='blue', label=par.outputstyle)
    plt.plot(np.arange(0, pred_out.shape[0]), pred_out[:, 0], linestyle='solid', color='red', label='prediction')
    plt.legend(loc='upper right')
    plt.show()
    if par.outputstyle == 'var':
        pd.DataFrame.from_records(
            {'logreturns': returns_var[:, 0],
             'var_n': pred_out[:, 0]}).to_csv(
            'cond_cov_n.csv', sep=',', index=False)
    else:
        pd.DataFrame.from_records({'logreturns': returns_var[:, 0], 'var_t': pred_out[:, 0]}).to_csv('cond_cov_t.csv', sep=',', index=False)
elif par.outputstyle == 'mean_sd':
    VaR_list = []
    VaR_t_list = []
    for i in range(len(par.alpha_all)):
        VaR = (pred_out[:, 0] + np.abs(pred_out[:, 1]) * np.sqrt((par.rolling_var + 1) / par.rolling_var) * t.ppf(par.alpha_all[i], par.rolling_var))
        VaR_t = (pred_out[:, 0] + np.abs(pred_out[:, 1]) * np.sqrt((par.dof - 2) / par.dof) * t.ppf(par.alpha_all[i], par.dof))
        noe_n = np.sum(returns_var[:, 0] - VaR < 0)
        noe_t = np.sum(returns_var[:, 0] - VaR_t < 0)
        print(noe_n / te_y_out.shape[0])
        print(noe_t / te_y_out.shape[0])
        plt.plot(np.arange(0, pred_out.shape[0]), returns_var[:, 0], linestyle='solid', color='red', label='logreturns')
        plt.plot(np.arange(0, pred_out.shape[0]), VaR, linestyle='solid', label='VaR')
        plt.title('VaR')
        plt.savefig('var_n_r_' + str(par.alpha_all[i]) + '.png')
        plt.show()
        plt.plot(np.arange(0, pred_out.shape[0]), returns_var[:, 0], linestyle='solid', color='red', label='logreturns')
        plt.plot(np.arange(0, pred_out.shape[0]), VaR_t, linestyle='solid', label='VaR_t')
        plt.title('VaR_t')
        plt.savefig('var_t_r_' + str(par.alpha_all[i]) + '.png')
        plt.show()
        plt.plot(np.arange(0, pred_out.shape[0]), te_y_out[:, 0], linestyle='solid', color='blue', label='mean')
        plt.plot(np.arange(0, pred_out.shape[0]), pred_out[:, 0], linestyle='solid', color='red', label='prediction')
        plt.title('meanpred_r')
        plt.legend(loc='upper right')
        plt.savefig('meanpred_r.png')
        plt.show()
        plt.plot(np.arange(0, pred_out.shape[0]), te_y_out[:, 1], linestyle='solid', color='blue', label='sd')
        plt.plot(np.arange(0, pred_out.shape[0]), np.abs(pred_out[:, 1]), linestyle='solid', color='red', label='prediction')
        plt.title('sdpred_r')
        plt.legend(loc='upper right')
        plt.savefig('sdpred_r.png')
        plt.show()
        VaR_list.append(VaR)
        VaR_t_list.append(VaR_t)
    VaR_out = np.vstack(VaR_list).transpose()
    VaR_t_out = np.vstack(VaR_t_list).transpose()
    pd.DataFrame.from_records({'logreturns': returns_var[:, 0],
                               'var_n_01': VaR_out[:, 0], 'var_n_025': VaR_out[:, 1], 'var_n_05': VaR_out[:, 2],
                               'var_n_1': VaR_out[:, 3], 'var_n_9': VaR_out[:, 4], 'var_n_95': VaR_out[:, 5],
                               'var_n_975': VaR_out[:, 6], 'var_n_99': VaR_out[:, 7],
                               'var_t_01': VaR_t_out[:, 0], 'var_t_025': VaR_t_out[:, 1], 'var_t_05': VaR_t_out[:, 2],
                               'var_t_1': VaR_t_out[:, 3], 'var_t_9': VaR_t_out[:, 4], 'var_t_95': VaR_t_out[:, 5],
                               'var_t_975': VaR_t_out[:, 6], 'var_t_99': VaR_t_out[:, 7], }
                              ).to_csv('cond_cov_r.csv', sep=',', index=False)

elif par.outputstyle == 'ts':
    print('No plots to show')
else:
    print('choose outputstyle from "mean_sd", "var", "var_t", "ts" ')

