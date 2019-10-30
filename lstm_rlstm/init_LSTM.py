import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import parameters as par
import numpy as np
import pandas as pd
import functions as fu
from scipy.stats import t
from keras_radam.training import RAdamOptimizer
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


load_scaler = False
num_epochs = 1
batch_size = 512
lr = 1e-3
# Data
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


input = keras.layers.Input(shape=fu.getinputshape_lstm(par.layer, par.n_steps, par.n_neu, par.n_res, features, par.reservoir))
l = keras.layers.Bidirectional(keras.layers.LSTM(par.lstm[0], return_sequences=False))(input)
l = keras.activations.tanh(l)
l = keras.layers.Dropout(0.2)(l)
l = keras.layers.Dense(par.dense[0])(l)
l = keras.activations.tanh(l)
l = keras.layers.Dropout(0.2)(l)
l = keras.layers.Dense(par.dense[1])(l)
l = keras.activations.tanh(l)
l = keras.layers.Dropout(0.2)(l)
l = keras.layers.Dense(par.dense[2], use_bias=False)(l)
l = keras.activations.tanh(l)
out = keras.layers.Dense(fu.getoutputshape_lstm(par.outputstyle, par.ts_out_length), activation='tanh', use_bias=False)(l)
model = keras.models.Model(inputs=input, outputs=out)
model.summary()
radam = RAdamOptimizer(learning_rate=lr)
model.compile(loss='mae', optimizer=radam, metrics=['mae'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs, verbose=1,
          validation_data=(test_x, test_y))
score, acc = model.evaluate(test_x, test_y, verbose=0)


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
    VaR = (pred[:, 0] + np.abs(pred[:, 1]) * np.sqrt((par.rolling_var + 1) / par.rolling_var) * t.ppf(par.alpha, par.rolling_var))
    VaR_t = (pred[:, 0] + np.abs(pred[:, 1]) * np.sqrt((par.dof - 2) / par.dof) * t.ppf(par.alpha, par.dof))
    noe_n = np.sum(returns_var_test[:, 0] - VaR < 0)
    noe_t = np.sum(returns_var_test[:, 0] - VaR_t < 0)
    print(noe_n / test_y.shape[0])
    print(noe_t / test_y.shape[0])
    plt.plot(np.arange(0, test_y.shape[0]), returns_var_test[:, 0], linestyle='solid', color='red', label='logreturns')
    plt.plot(np.arange(0, test_y.shape[0]), VaR, linestyle='solid', label='VaR')
    plt.title('VaR')
    plt.savefig('var_n_pred.png')
    plt.show()
    plt.plot(np.arange(0, test_y.shape[0]), returns_var_test[:, 0], linestyle='solid', color='red', label='logreturns')
    plt.plot(np.arange(0, test_y.shape[0]), VaR_t, linestyle='solid', label='VaR_t')
    plt.title('VaR_t')
    plt.savefig('var_t_pred.png')
    plt.show()
    plt.plot(np.arange(0, pred.shape[0]), test_y[:, 0], linestyle='solid', color='blue', label='mean')
    plt.plot(np.arange(0, pred.shape[0]), pred[:, 0], linestyle='solid', color='red', label='prediction')
    plt.title('meanpred')
    plt.legend(loc='upper right')
    plt.show()
    plt.plot(np.arange(0, pred.shape[0]), test_y[:, 1], linestyle='solid', color='blue', label='sd')
    plt.plot(np.arange(0, pred.shape[0]), np.abs(pred[:, 1]), linestyle='solid', color='red', label='prediction')
    plt.title('sdpred')
    plt.legend(loc='upper right')
    plt.show()
    pd.DataFrame.from_records({'logreturns': returns_var_test[:, 0], 'var_n': VaR, 'var_t': VaR_t}).to_csv(
        'cond_cov.csv', sep=',', index=False)
elif par.outputstyle == 'ts':
    print('No plots to show')
else:
    print('choose outputstyle from "mean_sd", "var", "var_t", "ts" ')

# save generated model
if par.rolling:
    model.save(par.model_name_rolling + '.h5')
    model.save_weights(par.model_name_rolling + '.weights.h5')
    print('Saved model to disk')
else:
    model.save(par.model_name + '.h5')
    model.save_weights(par.model_name + '.weights.h5')
    print('Saved model to disk')

