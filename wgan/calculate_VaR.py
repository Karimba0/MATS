import tensorflow.keras as keras
import functions as fu
import tensorflow as tf
import tqdm
import pandas as pd
import numpy as np
import parameters as par
from scipy.stats import t
import matplotlib.pyplot as plt
import joblib
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# parameters
scaler_filename = "scaler.save"
batch_size = 1

# Data
dates = pd.read_csv(par.data_name, delimiter=',')[['dt']]
if par.reservoir:
    data = np.load('reservoir.npz')
    train_x = data['train_x'].astype(np.float32)
    train_c = data['train_c'].astype(np.float32)
    train_y = data['train_y'].astype(np.float32)
    test_x = data['test_x'].astype(np.float32)
    test_y = data['test_y'].astype(np.float32)
else:
    data = pd.read_csv(par.data_name, delimiter=',')[['volume', 'RVOL', 'MACD', '20sd', 'High', 'Low', 'Adj Close',
                                                      'dt_diff', 'r_fft_100', 'r_fft_500', 'logreturns']].to_numpy()
    min_max_scaler = joblib.load(scaler_filename)
    data[:, :-1] = min_max_scaler.transform(data[:, :-1])
    data_close = pd.read_csv(par.data_name, delimiter=',')[['logreturns']].to_numpy()
    # First prediction index test set
    n_split = fu.get_datesplit(par.date, dates)
    train_x = data[:n_split - par.n_steps, :].astype(np.float32)
    train_c = data_close[:n_split - par.n_steps, :].astype(np.float32)
    train_y = data_close[par.n_steps:n_split, :].astype(np.float32)
    test_x = data[n_split - par.n_steps - par.ts_out_length + 1:- par.ts_out_length, :].astype(np.float32)
    test_y = data_close[n_split - par.ts_out_length + 1:, :].astype(np.float32)


dataset_test, dataset_test_x = fu.batch_dataset_test(test_x, test_y, par.reservoir, par.n_steps, par.ts_out_length,
                                                     batch_size=1, drop_remainder=True, n_prefetch_batch=1,
                                                     map_fn=False, n_map_threads=None, shuffle=True,
                                                     shuffle_buffer_size=None, repeat=1)


if par.reservoir:
    features = train_x.shape[2]
    len_dataset = (train_x.shape[0] // batch_size) * par.repeat
    len_dataset_test = test_y.shape[0] - par.ts_out_length + 1
else:
    features = train_x.shape[1]
    len_dataset = (train_x.shape[0] // batch_size) * par.repeat
    len_dataset_test = test_y.shape[0] - par.ts_out_length + 1

G = keras.models.load_model('G' + '.h5')
G.load_weights('G' + '.weights.h5')
G.summary()
out_pred = []
VaR_list = []
VaR_t_list = []
n = par.samples*par.ts_out_length
test_count = 0

# sequence length of input, needed for noise shape
if par.reservoir:
    second_dim_noise = par.n_input_layers * par.n_neu
else:
    second_dim_noise = par.n_steps


for te_x in tqdm.tqdm(dataset_test_x, desc='Test Loop', total=len_dataset_test):
    out_pred = []
    test_count += 1
    for i in range(par.samples):
        noise = tf.random.uniform(minval=-1, maxval=1, shape=[1, second_dim_noise, par.noise_size])
        te_x_sample = tf.concat([te_x, noise], axis=2)
        pred = G(te_x_sample, training=False)
        out_pred.append(pred)

    out_stack = np.hstack(out_pred)
    mean = np.mean(out_stack, axis=1)
    std = np.std(out_stack, axis=1)
    VaR = (mean + std*np.sqrt((n+1)/n) * t.ppf(par.alpha_all, n))
    VaR_t = (mean + std * np.sqrt((par.dof - 2) / par.dof) * t.ppf(par.alpha_all, par.dof))

    VaR_list.append(VaR)
    VaR_t_list.append(VaR_t)

print(test_count)
VaR_out = np.vstack(VaR_list)
VaR_t_out = np.vstack(VaR_t_list)
dates = dates.to_numpy()
for i in range(len(par.alpha_all)):
    var = VaR_out[:, i].reshape([-1, 1])
    nn_noe = np.sum((test_y[par.ts_out_length - 1:] - var < 0))
    print(nn_noe/(test_y.shape[0]-par.ts_out_length + 1))
    plt.plot(np.arange(0, test_y.shape[0]-par.ts_out_length + 1), test_y[par.ts_out_length - 1:], linestyle='solid',
             color='red', label='logreturns')
    plt.plot(np.arange(0, test_y.shape[0]-par.ts_out_length + 1), VaR_out[:, i], linestyle='solid', color='blue',
             label='VaR prediction')
    plt.legend(loc='upper right')
    plt.savefig('var_n_' + str(par.alpha_all[i]) + '.png')
    plt.show()
    plt.clf()
    var_t = VaR_t_out[:, i].reshape([-1, 1])
    nn_noe_t = np.sum((test_y[par.ts_out_length - 1:] - var_t < 0))
    plt.plot(np.arange(0, test_y.shape[0]-par.ts_out_length + 1), test_y[par.ts_out_length - 1:], linestyle='solid',
             color='red', label='logreturns')
    plt.plot(np.arange(0, test_y.shape[0]-par.ts_out_length + 1), VaR_t_out[:, i], linestyle='solid', color='blue',
             label='VaR_t prediction')
    plt.legend(loc='upper right')
    plt.savefig('var_t_' + str(par.alpha_all[i]) + '.png')
    plt.show()
    plt.clf()
    print(nn_noe_t/(test_y.shape[0]-par.ts_out_length + 1))
pd.DataFrame.from_records({'logreturns': test_y[par.ts_out_length - 1:, 0],
                               'var_n_01': VaR_out[:, 0], 'var_n_025': VaR_out[:, 1], 'var_n_05': VaR_out[:, 2],
                               'var_n_1': VaR_out[:, 3], 'var_n_9': VaR_out[:, 4], 'var_n_95': VaR_out[:, 5],
                               'var_n_975': VaR_out[:, 6], 'var_n_99': VaR_out[:, 7],
                               'var_t_01': VaR_t_out[:, 0], 'var_t_025': VaR_t_out[:, 1], 'var_t_05': VaR_t_out[:, 2],
                               'var_t_1': VaR_t_out[:, 3], 'var_t_9': VaR_t_out[:, 4], 'var_t_95': VaR_t_out[:, 5],
                               'var_t_975': VaR_t_out[:, 6], 'var_t_99': VaR_t_out[:, 7], }
                              ).to_csv('cond_cov.csv', sep=',', index=False)