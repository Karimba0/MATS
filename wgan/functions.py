import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parameters as par
from scipy.stats import t, norm


def compute_reservoir_wgan(hor, n_split, ts_out_length, n, n_neu, n_i, k_i, h, n_res, n_steps, data, data_y):
    train_x = data[:n_split - ts_out_length, :]
    test_x = data[n_split - n_steps - ts_out_length + 1:- ts_out_length, :]
    # number of teaching signals
    t_star = train_x.shape[0] // n_i
    # number of testing Signals
    t_star_test = test_x.shape[0] // n_i
    # input matrix
    w_in = np.random.uniform(low=-0.1, high=0.1, size=(k_i, n, n_res))

    # allocation train
    I = np.zeros([n_neu, t_star, n_res])
    u = np.zeros([k_i, n_split - ts_out_length, n_res])
    x = np.zeros([n_neu, t_star, n_res])

    # allocation test
    I_test = np.zeros([n_neu, t_star_test, n_res])
    u_test = np.zeros([k_i, test_x.shape[0], n_res])
    x_test = np.zeros([n_neu, t_star_test, n_res])
    pred = np.zeros([test_x.shape[0], n, n_res])
    error = np.zeros([n_res])
    # training parameters
    p = 2
    nu = np.random.uniform(low=0.8, high=1.5, size=n_res)
    save_nu = nu
    mu = np.random.uniform(low=1.2, high=2, size=n_res)
    save_mu = mu
    lamb = np.random.uniform(low=5, high=10, size=1)
    save_lamb = lamb
    # run reservoirs for train and test
    for i in range(n_res):
        u[:, :, i] = gen_u(train_x, w_in[:, :, i])
        I[:, :, i] = gen_I(u[:, :, i], t_star)
        x[:, :, i] = gen_x(x[:, :, i], I[:, :, i], n_neu, h, t_star, nu[i], mu[i], p)
        u_test[:, :, i] = gen_u(test_x, w_in[:, :, i])
        I_test[:, :, i] = gen_I(u_test[:, :, i], t_star_test)
        x_test[:, :, i] = gen_x(x_test[:, :, i], I_test[:, :, i], n_neu, h, t_star_test, nu[i], mu[i], p)
        print(i)

    # Data preparation
    x = x.swapaxes(0, 1)
    x_test = x_test.swapaxes(0, 1)

    train_x = data_prep_tdr_lstm(x, n_steps, t_star, n_neu, n_res)
    train_c = data_y[:n_split - n_steps, :]
    train_y = data_y[n_steps:n_split, :]
    test_x = data_prep_tdr_lstm(x_test, n_steps, t_star_test, n_neu, n_res)
    test_y = data_y[n_split - ts_out_length + 1:, :]
    #train_y, test_y = ydatadiff(train_y, test_y)
    np.savez_compressed('Reservoir', train_x=train_x, train_c=train_c, train_y=train_y, test_x=test_x, test_y=test_y)
    np.save('train_x.npy', train_x)
    np.save('train_c.npy', train_c)
    np.save('train_y.npy', train_y)
    np.save('test_x.npy', test_x)
    np.save('test_y.npy', test_y)
    save = np.stack((error, save_mu, save_nu), axis=1)
    np.savetxt("values.csv", save, delimiter=',', header='error,mu,nu', comments='')
    return None


def data_prep_tdr_lstm(x, n_steps, t_star, n_neu, n_res):
    x_temp = np.zeros([t_star-n_steps+1, n_neu*n_steps, n_res])
    for i in range(n_res):
        x_temp[:, :, i] = window_stack(x[:, :, i], 1, n_steps)
    return x_temp


def gen_u(data, w_in):
    u = np.matmul(w_in, data.transpose())
    return u


def gen_I(u,t_star):
    I = u.transpose().reshape(t_star, -1).transpose()
    return I


def gen_x(x, I, n_neu, h, t_star, nu, mu, p):
    for k in range(t_star):
        if k == 0:
            x[:, k] = I[:, k]
        else:
            x[0, k] = np.exp(-h) * x[n_neu - 1, k - 1] + (1 - np.exp(-h)) * functional(x[0, k - 1], I[0, k], nu, mu, p)
            for i in range(1, n_neu, 1):
                x[i, k] = np.exp(-h) * x[i - 1, k] + (1 - np.exp(-h)) * functional(x[i, k - 1], I[i, k], nu, mu, p)
    return x


def gen_w_out(x, lamb, n_neu, y):
    w_out = np.matmul(np.linalg.inv(np.add(np.matmul(x, np.transpose(x)), lamb*np.eye(N=n_neu, M=n_neu))), np.matmul(x, y))
    return w_out


def functional(x, I, nu, mu, p):
    z = (nu*(x+mu*I))/(1+(x+mu*I)**p)
    return z


def prepx(x, xt, n_steps, layer):
    features = x.shape[1]
    if layer == 'dense':
        return x, xt, features
    else:
        x_seq = np.zeros([(x.shape[0]-n_steps), n_steps, features])
        xt_seq = np.zeros([(xt.shape[0]-n_steps), n_steps, features])
        for i in range(x.shape[0]-n_steps):
            x_seq[i, :, :] = x[i:(i+n_steps), :]
            if i <= xt.shape[0]-n_steps-1:
                xt_seq[i, :, :] = xt[i:(i+n_steps), :]

        return x_seq, xt_seq, features


def get_datesplit(date, data):
    n_split = data[data['dt'] == date].index.values.astype(int)[0]
    return n_split

# XGBoost
def get_data_ind(data, include_indicators=True, include_fft=True):
    if include_indicators:
        data = get_technical_indicators(data)
    if include_fft:
        data = get_fourier_transforms(data)

    return data


def get_feature_importance_data(data, data_y, outputstyle):
    data = data.copy()
    if outputstyle == 'var':
        y = data_y['var']
    elif outputstyle == 'var_t':
        y = data_y['var_t']
    elif outputstyle == 'mean':
        y = data_y['mean']
    elif outputstyle == 'std':
        y = data_y['std']
    else:
        y = data_y['logreturns']
    x = data.iloc[:, 1:]

    train_samples = int(x.shape[0] * 0.65)

    x_train = x.iloc[:train_samples]
    x_test = x.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (x_train, y_train), (x_test, y_test)


def get_fourier_transforms(data):
    data_ft = data[['dt', 'close', 'logreturns']]
    close_fft = np.fft.fft(np.asarray(data_ft['close'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(21, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [5, 10, 100, 1000]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        data['fft_{}'.format(num_)] = np.real(np.fft.ifft(fft_list_m10))
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(data_ft['close'], label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('close prices and fourier transforms')
    plt.legend()
    plt.savefig('fourier_close.png')
    #plt.show()

    logreturn_fft = np.fft.fft(np.asarray(data_ft['logreturns'].tolist()))
    r_fft_df = pd.DataFrame({'fft': logreturn_fft})
    r_fft_df['absolute'] = r_fft_df['fft'].apply(lambda x: np.abs(x))
    r_fft_df['angle'] = r_fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(21, 7), dpi=100)
    r_fft_list = np.asarray(r_fft_df['fft'].tolist())
    for num_ in [100, 500, 1000, 1500]:
        r_fft_list_m10 = np.copy(r_fft_list)
        r_fft_list_m10[num_:-num_] = 0
        data['r_fft_{}'.format(num_)] = np.real(np.fft.ifft(r_fft_list_m10))
        plt.plot(np.fft.ifft(r_fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(data_ft['logreturns'], label='Real')
    plt.xlabel('Days')
    plt.ylabel('logreturns')
    plt.title('logreturns and fourier transforms')
    plt.legend()
    plt.savefig('fourier_return.png')
    #plt.show()
    return data


def get_technical_indicators(data):
    # 7 and 21 days Moving Average
    data['ma7'] = data['close'].rolling(window=7).mean()
    data['ma21'] = data['close'].rolling(window=21).mean()

    # MACD
    data['26ema'] = data['close'].ewm(span=26).mean()
    data['12ema'] = data['close'].ewm(span=12).mean()
    data['MACD'] = (data['12ema'] - data['26ema'])

    # Bollinger Bands
    data['20sd'] = data['close'].rolling(20).std()
    data['upper_band'] = data['ma21'] + (data['20sd'] * 2)
    data['lower_band'] = data['ma21'] - (data['20sd'] * 2)

    # Exponential moving average
    data['ema'] = data['close'].ewm(com=0.5).mean()

    # Momentum
    data['momentum'] = data['close'] - 1

    # VaR calculation
    data['mean'] = data['logreturns'].rolling(par.rolling_var).mean()
    data['std'] = data['logreturns'].rolling(par.rolling_var).std(ddof=1)
    data['var'] = -(data['mean'] + data['std'] *
                    np.sqrt((par.rolling_var + 1) / par.rolling_var) * t.ppf(par.alpha, par.rolling_var))
    data['var_t'] = -(data['mean'] + data['std'] * np.sqrt((par.dof - 2) / par.dof) * t.ppf(par.alpha, par.dof))
    return data

# Data prep
def batch_dataset_train(input_G_ts, input_D_ts, train_y, batch_size, reservoir, n_steps, ts_out_length,
                        drop_remainder=True, n_prefetch_batch=None, n_map_threads=None, shuffle=True,
                        shuffle_buffer_size=None, repeat=None):

    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 8, 64)
    if reservoir:
        input_G_ts = tf.data.Dataset.from_tensor_slices(input_G_ts)
        input_D_ts = tf.data.Dataset.from_tensor_slices(input_D_ts)
        train_y = tf.data.Dataset.from_tensor_slices(train_y)
        input_D_ts = input_D_ts.window(n_steps, 1, 1).flat_map(
            lambda x: x.batch(n_steps, drop_remainder=drop_remainder))
        train_y = train_y.window(ts_out_length, 1, 1).flat_map(
            lambda x: x.batch(ts_out_length, drop_remainder=drop_remainder))
        data = tf.data.Dataset.zip((input_G_ts, input_D_ts, train_y))
        data = data.shuffle(shuffle_buffer_size).batch(batch_size, drop_remainder=drop_remainder)
        data = data.repeat(repeat).prefetch(n_prefetch_batch)
    else:
        input_G_ts = tf.data.Dataset.from_tensor_slices(input_G_ts)
        input_D_ts = tf.data.Dataset.from_tensor_slices(input_D_ts)
        train_y = tf.data.Dataset.from_tensor_slices(train_y)
        input_G_ts = input_G_ts.window(n_steps, 1, 1).flat_map(lambda x: x.batch(n_steps, drop_remainder=drop_remainder))
        input_D_ts = input_D_ts.window(n_steps, 1, 1).flat_map(lambda x: x.batch(n_steps, drop_remainder=drop_remainder))
        train_y = train_y.window(ts_out_length, 1, 1).flat_map(lambda x: x.batch(ts_out_length, drop_remainder=drop_remainder))
        data = tf.data.Dataset.zip((input_G_ts, input_D_ts, train_y))
        data = data.cache().shuffle(shuffle_buffer_size).batch(batch_size, drop_remainder=drop_remainder)
        data = data.repeat(repeat)
        # can put prefetch batch at the end again but doesnt help
    return data


def batch_dataset_test(input_G_ts, train_y, reservoir, n_steps, ts_out_length, batch_size, drop_remainder=True, n_prefetch_batch=None, map_fn=None,
                  n_map_threads=None, shuffle=True, shuffle_buffer_size=None, repeat=None):

    if reservoir:
        input_G_ts = tf.data.Dataset.from_tensor_slices(input_G_ts)
        train_y = tf.data.Dataset.from_tensor_slices(train_y)
        train_y = train_y.window(ts_out_length, 1, 1).flat_map(
            lambda x: x.batch(ts_out_length, drop_remainder=drop_remainder))
        data = tf.data.Dataset.zip((input_G_ts, train_y))
        data = data.cache().batch(batch_size)
        data_x = input_G_ts.cache().batch(batch_size)
    else:
        input_G_ts = tf.data.Dataset.from_tensor_slices(input_G_ts)
        train_y = tf.data.Dataset.from_tensor_slices(train_y)
        input_G_ts = input_G_ts.window(n_steps, 1, 1).flat_map(lambda x: x.batch(n_steps, drop_remainder=drop_remainder))
        train_y = train_y.window(ts_out_length, 1, 1).flat_map(lambda x: x.batch(ts_out_length, drop_remainder=drop_remainder))
        data = tf.data.Dataset.zip((input_G_ts, train_y))
        data = data.cache().batch(batch_size)
        data_x = input_G_ts.cache().batch(batch_size)
    return data, data_x


def prepdata(x, xt, n_steps, layer):
    features = x.shape[1]
    if layer == 'dense':
        if n_steps == 1:
            return x, xt, features
        else:
            x = rolling_window(x.reshape(1, -1), n_steps).reshape(-1, n_steps*features)
            xt = rolling_window(xt.reshape(1, -1), n_steps).reshape(-1, n_steps*features)
            return x, xt, features

    else:
        x = rolling_window(x, n_steps)
        xt = rolling_window(xt, n_steps)
        return x, xt, features


# Losses
def get_gan_losses_fn():
    bce = tf.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(tf.ones_like(r_logit), r_logit)
        f_loss = bce(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(- f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = tf.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()


def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = alpha * a + (1-alpha) * b
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(tf.reshape(grad, [tf.shape(grad)[0], -1]))))
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)

    return gp


# Input and output Shapes
def getoutputshape_G(outputstyle, ts_out_length):
    if outputstyle == 'mean_sd':
        return 2
    elif outputstyle == 'ts':
        return ts_out_length
    else:
        print('choose "call", "put" or "both"')


def getinputshape_G(layer, n_steps, n_neu, n_res, features, reservoir, noise_size):
    if reservoir:
        if layer == 'lstm':
            return [n_steps*n_neu, n_res + noise_size]
        elif layer == 'dense':
            return [features]
    else:
        if layer == 'lstm':
            return [n_steps, features + noise_size]
        elif layer == 'dense':
            return [features]


def getinputshape_C(outputstyle, layer, n_steps, n_neu, features, ts_out_length, reservoir):
    if outputstyle == 'mean_sd':
        if reservoir:
            if layer == 'dense':
                return [n_steps, features]
            elif layer in ['convolution', 'lstm']:
                return [n_steps + ts_out_length, features]
        else:
            if layer == 'dense':
                return [n_steps, features]
            elif layer in ['convolution', 'lstm']:
                return [n_steps + ts_out_length, features]
    else:
        if reservoir:
            if layer == 'dense':
                return [n_steps + ts_out_length]
            elif layer in ['convolution', 'lstm']:
                return [n_steps + ts_out_length, 1]
        else:
            if layer == 'dense':
                return [n_steps + ts_out_length]
            elif layer in ['convolution', 'lstm']:
                return [n_steps + ts_out_length, 1]


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


# Build Networks
# LSTM only generator
def build_G_L(layer, n_steps, noise_size, n_neu, n_res, features, outputstyle, ts_out_length, reservoir, lstm, name='Generator'):
    input = keras.layers.Input(shape=getinputshape_G(layer, n_steps, n_neu, n_res, features, reservoir, noise_size))
    l = keras.layers.Bidirectional(keras.layers.LSTM(lstm[0], return_sequences=False))(input)
    l = keras.activations.tanh(l)
    out = keras.layers.Dense(getoutputshape_G(outputstyle, ts_out_length), activation='tanh')(l)

    return keras.Model(inputs=input, outputs=out, name=name)

# LSTM with dense generator
def build_G_LD(layer, n_steps, noise_size, n_neu, n_res, features, outputstyle, ts_out_length, reservoir, lstm, dense, name='Generator'):
    input = keras.layers.Input(shape=getinputshape_G(layer, n_steps, n_neu, n_res, features, reservoir, noise_size))
    l = keras.layers.Bidirectional(keras.layers.LSTM(lstm[0], return_sequences=False))(input)
    l = keras.activations.tanh(l)
    l = keras.layers.Dropout(0.2)(l)
    l = keras.layers.Dense(dense[0])(l)
    l = keras.activations.tanh(l)
    l = keras.layers.Dropout(0.2)(l)
    #l = keras.layers.Dense(dense[1])(l)
    #l = keras.activations.tanh(l)
    #l = keras.layers.Dropout(0.2)(l)
    #l = keras.layers.Dense(dense[2], use_bias=False)(l)
    #l = keras.activations.tanh(l)
    out = keras.layers.Dense(getoutputshape_G(outputstyle, ts_out_length), activation='tanh')(l)

    return keras.Model(inputs=input, outputs=out, name=name)

# Fully connected Critic
def build_C_D(outputstyle, layer, n_steps, n_neu, features, ts_out_length, reservoir, dense, name='DenseDiscriminator'):
    input = keras.layers.Input(shape=getinputshape_C(outputstyle, layer, n_steps, n_neu, features, ts_out_length, reservoir))
    l = keras.layers.Dense(dense[0])(input)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Dropout(0.2)(l)
    l = keras.layers.Dense(dense[1])(l)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Dense(dense[2])(l)
    l = keras.layers.Dense(dense[3], use_bias=False)(l)
    l = keras.layers.ReLU()(l)
    out = keras.layers.Dense(1, use_bias=False)(l)
    return keras.Model(inputs=input, outputs=out, name=name)

# LSTM Critic (not working yet because of internal keras optimization problem)
def build_C_L(outputstyle, layer, n_steps, n_neu, features, ts_out_length, reservoir, dense, lstm_C, name='DenseDiscriminator'):
    input = keras.layers.Input(shape=getinputshape_C(outputstyle, layer, n_steps, n_neu, features, ts_out_length, reservoir))
    l = keras.layers.LSTM(lstm_C[0], return_sequences=False)(input)
    l = keras.activations.tanh(l)
    l = keras.layers.Dense(dense[0])(l)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Dense(dense[1])(l)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Dense(dense[2], use_bias=False)(l)
    l = keras.layers.ReLU()(l)
    out = keras.layers.Dense(1, use_bias=False)(l)
    return keras.Model(inputs=input, outputs=out, name=name)

# Convolutional Critic
def build_C_CD(outputstyle, layer, n_steps, n_neu, features, ts_out_length, reservoir, convolution, dense, name='ConvDiscriminator'):
    input = keras.layers.Input(shape=getinputshape_C(outputstyle, layer, n_steps, n_neu, features, ts_out_length, reservoir))
    l = keras.layers.Conv1D(convolution[0], 5, strides=2)(input)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Conv1D(convolution[1], 5, strides=2)(l)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Conv1D(convolution[2], 5, strides=2)(l)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Dense(dense[0], use_bias=False)(l)
    l = keras.layers.LeakyReLU(alpha=0.01)(l)
    l = keras.layers.Dense(dense[1], use_bias=False)(l)
    l = keras.layers.ReLU()(l)
    out = keras.layers.Dense(1)(l)

    return keras.Model(inputs=input, outputs=out, name=name)


# Helper functions
def changedate(my_data):
    dates = my_data[:, 0]
    datesdiff = dates[0:dates.shape[0]-1, ]-dates[1:dates.shape[0], ]
    return datesdiff


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_stack(a, stepsize=1, width=3):
    return np.hstack(a[i:1+i-width or None:stepsize] for i in range(0, width))


# RVOL calculation
def calc_rvol(data, alpha, n, trading_days):
    high = data['high'].to_numpy()
    low = data['low'].to_numpy()
    open = data['open'].to_numpy()
    close = data['close'].to_numpy()
    o = pd.Series(np.subtract(np.log(open[1:]), np.log(close[:-1])))
    u = pd.Series(np.subtract(np.log(high[1:]), np.log(open[1:])))
    d = pd.Series(np.subtract(np.log(low[1:]), np.log(open[1:])))
    c = pd.Series(np.subtract(np.log(close[1:]), np.log(open[1:])))
    rs = pd.Series(np.add(np.multiply(u, np.subtract(u, c)), np.multiply(d, np.subtract(d, c))))
    V_rs = trading_days * (rs.rolling(window=n).mean().to_numpy())
    V_0 = trading_days * (o.rolling(window=n).var().to_numpy())
    V_c = trading_days * (c.rolling(window=n).var().to_numpy())
    k = (alpha-1)/(alpha+((n+1)/(n-1)))
    V = np.sqrt(np.add(V_0, np.add(k * V_c, (1-k) * V_rs)))
    return V
