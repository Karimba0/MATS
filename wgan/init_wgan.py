import tensorflow.keras as keras
import functions as fu
import functools
import tensorflow as tf
import tqdm
import pandas as pd
import numpy as np
import parameters as par
from sklearn import preprocessing
from keras_radam.training import RAdamOptimizer
import joblib
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# parameters
batch_size = 512
epochs = 10
lr = 1e-3
# beta_1 for radam or adam optimizer
beta_1 = 0.95
# critic updates per generator update
n_critic = 5
adversarial_loss_mode = 'wgan'
gradient_penalty_mode = 'wgan-gp'
gradient_penalty_weight = 10.0
scaler_filename = "scaler.save"

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
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data[:, :-1] = min_max_scaler.fit_transform(data[:, :-1])
    joblib.dump(min_max_scaler, scaler_filename)
    data_close = pd.read_csv(par.data_name, delimiter=',')[['logreturns']].to_numpy()
    # First prediction index test set
    n_split = fu.get_datesplit(par.date, dates)
    train_x = data[:n_split - par.ts_out_length, :].astype(np.float32)
    train_c = data_close[:n_split - par.ts_out_length, :].astype(np.float32)
    train_y = data_close[par.n_steps:n_split, :].astype(np.float32)
    test_x = data[n_split - par.n_steps - par.ts_out_length + 1:- par.ts_out_length, :].astype(np.float32)
    test_y = data_close[n_split - par.ts_out_length + 1:, :].astype(np.float32)


dataset_train = fu.batch_dataset_train(train_x, train_c, train_y, batch_size, par.reservoir, par.n_steps,
                                       par.ts_out_length, drop_remainder=True, n_prefetch_batch=par.prefetch_batch,
                                       n_map_threads=None, shuffle=True, shuffle_buffer_size=None, repeat=par.repeat)
dataset_test, dataset_test_x = fu.batch_dataset_test(test_x, test_y, par.reservoir, par.n_steps, par.ts_out_length,
                                                     batch_size=1, drop_remainder=True, n_prefetch_batch=1,
                                                     map_fn=False, n_map_threads=None, shuffle=True,
                                                     shuffle_buffer_size=None, repeat=1)

if par.reservoir:
    features = train_x.shape[2]
    len_dataset = (train_x.shape[0] // batch_size) * par.repeat
    len_dataset_test = test_x.shape[0]
else:
    features = train_x.shape[1]
    len_dataset = (train_x.shape[0] // batch_size) * par.repeat
    len_dataset_test = test_x.shape[0]

# networks
if par.G_layer == 'l':
    G = fu.build_G_L(par.layer, par.n_steps, par.noise_size, par.n_neu, par.n_res, features, par.outputstyle,
                   par.ts_out_length, par.reservoir, par.lstm_G, name='Generator_l')
elif par.G_layer == 'ld':
    G = fu.build_G_LD(par.layer, par.n_steps, par.noise_size, par.n_neu, par.n_res, features, par.outputstyle,
                   par.ts_out_length, par.reservoir, par.lstm_G, par.dense_G, name='Generator_ld')
if par.C_layer == 'dense':
    D = fu.build_C_D(par.outputstyle, par.C_layer, par.n_steps, par.n_neu, features, par.ts_out_length,
                     par.reservoir, par.dense_C, name='Dense_Discriminator')
elif par.C_layer == 'lstm':
    D = fu.build_C_L(par.outputstyle, par.C_layer, par.n_steps, par.n_neu, features, par.ts_out_length,
                     par.reservoir, par.dense_C, par.lstm_C, name='LSTM_Discriminator')
elif par.C_layer == 'convolution':
    D = fu.build_C_CD(par.outputstyle, par.C_layer, par.n_steps, par.n_neu, features, par.ts_out_length,
                      par.reservoir, par.conv_C, par.dense_C_C, name='Convolutional_Discriminator')
else:
    print('choose ')
G.summary()
D.summary()

# adversarial_loss_functions
d_loss_fn, g_loss_fn = fu.get_adversarial_losses_fn(adversarial_loss_mode)
# Adam
if par.optimizer == 'adam':
    G_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
    D_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
# RAdam
if par.optimizer == 'radam':
    G_optimizer = RAdamOptimizer(learning_rate=lr)
    D_optimizer = RAdamOptimizer(learning_rate=lr)


@tf.function
def train_G(tr_x, tr_x_D, batch_size, n_steps, C_layer):
    with tf.GradientTape() as t:
        x_G = G(tr_x, training=True)
        if C_layer == 'dense':
            tr_x_D = tf.reshape(tr_x_D, [batch_size, n_steps])
        else:
            x_G = tf.reshape(x_G, [batch_size, -1, 1])
        x_G = tf.concat([tr_x_D, x_G], axis=1)
        x_fake_d_logit = D(x_G, training=True)
        G_loss = g_loss_fn(x_fake_d_logit)

    G_grad = t.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

    return {'g_loss': G_loss}


@tf.function
def train_D(tr_x, tr_x_D, tr_y, batch_size, n_steps, ts_out_length, C_layer):
    with tf.GradientTape() as t:
        x_G = G(tr_x, training=True)
        if C_layer == 'dense':
            tr_x_D = tf.reshape(tr_x_D, [batch_size, n_steps])
            tr_y = tf.reshape(tr_y, [batch_size, ts_out_length])
        else:
            x_G = tf.reshape(x_G, [batch_size, -1, 1])
        x_G = tf.concat([tr_x_D, x_G], axis=1)
        tr_y = tf.concat([tr_x_D, tr_y], axis=1)
        x_real_d_logit = D(tr_y, training=True)
        x_fake_d_logit = D(x_G, training=True)

        x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
        gp = fu.gradient_penalty(functools.partial(D, training=True), tr_y, x_G, mode=gradient_penalty_mode)

        D_loss = (x_real_d_loss + x_fake_d_loss) + gp * gradient_penalty_weight

    D_grad = t.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


@tf.function
def sample(z):
    return G(z, training=False)


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

# sequence length of input, needed for noise shape
if par.reservoir:
    second_dim_noise = par.n_input_layers * par.n_neu
else:
    second_dim_noise = par.n_steps


if par.optimizer == 'adam':
    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        train_step = 0
        for tr_x, tr_x_D, tr_y in tqdm.tqdm(dataset_train, desc='Inner Epoch Loop', total=len_dataset):
            noise = tf.random.uniform(minval=-1, maxval=1, shape=[batch_size, second_dim_noise, par.noise_size])
            tr_x = tf.concat([tr_x, noise], axis=2)
            D_loss_dict = train_D(tr_x, tr_x_D, tr_y, batch_size, par.n_steps, par.ts_out_length, par.C_layer)

            if D_optimizer.iterations.numpy() % n_critic == 0:
                G_loss_dict = train_G(tr_x, tr_x_D, batch_size, par.n_steps, par.C_layer)

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                x_fake = sample(tr_x)

    for te_x, te_y in tqdm.tqdm(dataset_test, desc='Test Loop', total=len_dataset_test):
        noise = tf.random.uniform(minval=-1, maxval=1, shape=[1, second_dim_noise, par.noise_size])
        te_x = tf.concat([te_x, noise], axis=2)
        pred = G(te_x)


if par.optimizer == 'radam':
    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        train_step = 0
        for tr_x, tr_x_D, tr_y in tqdm.tqdm(dataset_train, desc='Inner Epoch Loop', total=len_dataset):
            train_step += 1
            noise = tf.random.uniform(minval=-1, maxval=1, shape=[batch_size, second_dim_noise, par.noise_size])
            tr_x = tf.concat([tr_x, noise], axis=2)
            D_loss_dict = train_D(tr_x, tr_x_D, tr_y, batch_size, par.n_steps, par.ts_out_length, par.C_layer)

            if train_step % n_critic == 0:
                G_loss_dict = train_G(tr_x, tr_x_D, batch_size, par.n_steps, par.C_layer)

            # sample
            if train_step % 100 == 0:
                x_fake = sample(tr_x)

    for te_x, te_y in tqdm.tqdm(dataset_test, desc='Test Loop', total=len_dataset_test):
        noise = tf.random.uniform(minval=-1, maxval=1, shape=[1, second_dim_noise, par.noise_size])
        te_x = tf.concat([te_x, noise], axis=2)
        pred = G(te_x)

print(pred)
print(np.subtract(pred, te_y).shape)
print(np.subtract(pred, te_y))

G.save('G' + ".h5")
G.save_weights('G' + '.weights.h5')
D.save('C' + ".h5")
D.save_weights('C' + '.weights.h5')
print("Saved model to disk")
