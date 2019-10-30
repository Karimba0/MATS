import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
from scipy.stats import t


def get_data_raw(data, data_y, outputstyle, n_split, rolling_var, n_steps, ts_out_length, hor, alpha, layer, reservoir,
                 dof, load_scaler):
    data = data.to_numpy()
    if outputstyle == 'ts':
        scaler_filename = "scaler_ts.save"
        train_x = data[:n_split - ts_out_length, :]
        test_x = data[n_split - n_steps:-ts_out_length, :]
        data_y = data_y.to_numpy()
        train_y = data_y[n_steps:n_split, 0]
        test_y = data_y[n_split:, 0]
        train_y = rolling_window(train_y, ts_out_length)
        test_y = rolling_window(test_y, ts_out_length)
        if reservoir:
            return train_x, test_x, train_y, test_y

        else:
            if load_scaler:
                min_max_scaler = joblib.load(scaler_filename)
                train_x[:, :-1] = min_max_scaler.fit_transform(train_x[:, :-1])
                test_x[:, :-1] = min_max_scaler.transform(test_x[:, :-1])
            else:
                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                train_x[:, :-1] = min_max_scaler.fit_transform(train_x[:, :-1])
                test_x[:, :-1] = min_max_scaler.transform(test_x[:, :-1])
                joblib.dump(min_max_scaler, scaler_filename)
            train_x, test_x, features = prepdata(train_x, test_x, n_steps, layer)
            return train_x, test_x, train_y, test_y, features

    elif outputstyle in ['mean_sd', 'var', 'var_t']:
        scaler_filename = "scaler.save"
        if load_scaler:
            min_max_scaler = joblib.load(scaler_filename)
            data[:, :-1] = min_max_scaler.transform(data[:, :-1])
        else:
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            data[:, :-1] = min_max_scaler.fit_transform(data[:, :-1])
            joblib.dump(min_max_scaler, scaler_filename)
        train_x = data[max(0, rolling_var - n_steps - hor):n_split - hor, :]
        test_x = data[n_split - n_steps - hor + 1:-hor, :]
        data_y['mean'] = data_y.rolling(rolling_var).mean()
        data_y['std'] = data_y['logreturns'].rolling(rolling_var).std(ddof=1)
        data_y['var'] = (data_y['mean'] + data_y['std'] * np.sqrt((rolling_var + 1) / rolling_var) *
                         t.ppf(alpha, rolling_var))
        data_y['var_t'] = (data_y['mean'] + data_y['std'] * np.sqrt((dof - 2) / dof) * t.ppf(alpha, dof))
        returns_var_test = data_y[['logreturns']].to_numpy()[n_split:]

        if outputstyle == 'var':
            data_y = data_y[['var']].to_numpy()
        elif outputstyle == 'mean_sd':
            data_y = data_y[['mean', 'std']].to_numpy()
        elif outputstyle == 'var_t':
            data_y = data_y[['var_t']].to_numpy()
        else:
            print('choose outputstyle from "mean_sd", "var", "var_t", "ts" ')
        train_y = data_y[max(0, rolling_var - n_steps - hor) + hor + n_steps - 1:n_split, :]
        test_y = data_y[n_split:, :]
        scaler_filename = "scaler.save"
        if reservoir:
            return train_x, test_x, train_y, test_y, returns_var_test
        else:
            train_x, test_x, features = prepdata(train_x, test_x, n_steps, layer)
            return train_x, test_x, train_y, test_y, features, returns_var_test
    else:
        print('choose outputstyle from "mean_sd", "var", "ts" ')


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
        x = window_stack(x, 1, n_steps).reshape([-1, n_steps, features])
        xt = window_stack(xt, 1, n_steps).reshape([-1, n_steps, features])
        return x, xt, features


def getoutputshape_lstm(outputstyle, ts_out_length):
    if outputstyle == 'mean_sd':
        return 2
    elif outputstyle == 'ts':
        return ts_out_length
    elif outputstyle in ['var', 'var_t']:
        return 1
    else:
        print('choose outputstyle from "mean_sd", "var", "ts" ')


def getinputshape_lstm(layer, n_steps, n_neu, n_res, features, reservoir):
    if reservoir:
        if layer == 'lstm':
            return [n_steps*n_neu, n_res]
        elif layer == 'dense':
            return [features]
    else:
        if layer == 'lstm':
            return [n_steps, features]
        elif layer == 'dense':
            return [features]


# Helper functions
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_datesplit(date, data):
    n_split = data[data['dt'] == date].index.values.astype(int)[0]
    return n_split


def ydatadiff(train_y, test_y):
    traindiff = train_y[1:train_y.shape[0], ] - train_y[0:train_y.shape[0] - 1, ]
    testdiff = test_y[1:test_y.shape[0], ] - test_y[0:test_y.shape[0] - 1, ]
    return traindiff, testdiff


def changedate(my_data):
    dates = my_data[:, 0]
    datesdiff = dates[0:dates.shape[0]-1, ]-dates[1:dates.shape[0], ]
    return datesdiff


def window_stack(a, stepsize=1, width=3):
    return np.hstack(a[i:1+i-width or None:stepsize] for i in range(0, width))


# RVOL calculation
def calc_rvol(data, alpha, n, trading_days):
    high, low, open, close = data.transpose()
    o = pd.Series(np.subtract(np.log(open[1:]), np.log(close[0:close.shape[0]-1])))
    u = pd.Series(np.subtract(np.log(high[1:]), np.log(open[1:])))
    d = pd.Series(np.subtract(np.log(low[1:]), np.log(open[1:])))
    c = pd.Series(np.subtract(np.log(close[1:]), np.log(open[1:])))
    rs = pd.Series(np.add(np.multiply(u, np.subtract(u, c)), np.multiply(d, np.subtract(d, c))))
    v_rs = trading_days * (rs.rolling(window=n).mean().to_numpy())
    v_0 = trading_days * (o.rolling(window=n).var().to_numpy())
    v_c = trading_days * (c.rolling(window=n).var().to_numpy())
    k = (alpha-1)/(alpha+((n+1)/(n-1)))
    v = np.sqrt(np.add(v_0, np.add(k * v_c, (1-k) * v_rs)))
    return v


# Reservoir functions
def compute_reservoir_lstm(data, data_y, outputstyle, n_split, rolling_var, n_steps, ts_out_length, hor, alpha, layer, reservoir, dof, k_i, n_i, n, n_res, h, n_neu):
    #data
    if outputstyle in ['mean_sd', 'var', 'var_t']:
        train_x, test_x, train_y, test_y, returns_var_test = get_data_raw(data, data_y, outputstyle, n_split, rolling_var, n_steps, ts_out_length, hor, alpha, layer, reservoir, dof)
    elif outputstyle == 'ts':
        train_x, test_x, train_y, test_y = get_data_raw(data, data_y, outputstyle, n_split, rolling_var, n_steps, ts_out_length, hor, alpha, layer, reservoir, dof)
    else:
        print('choose outputstyle from "mean_sd", "var", "var_t", "ts" ')

    # number of teaching signals
    t_star = train_x.shape[0] // n_i

    # number of testing Signals
    t_star_test = test_x.shape[0] // n_i

    # input matrix
    w_in = np.random.uniform(low=-0.1, high=0.1, size=(k_i, n, n_res))

    # allocation train
    I = np.zeros([n_neu, t_star, n_res])
    u = np.zeros([k_i, train_x.shape[0], n_res])
    x = np.zeros([n_neu, t_star, n_res])

    # allocation test
    I_test = np.zeros([n_neu, t_star_test, n_res])
    u_test = np.zeros([k_i,  test_x.shape[0], n_res])
    x_test = np.zeros([n_neu, t_star_test, n_res])
    pred = np.zeros([ test_x.shape[0], n, n_res])
    error = np.zeros([n_res])
    # training parameters
    p = 1
    nu = np.random.uniform(low=0.8, high=1.5, size=n_res)
    save_nu = nu
    mu = np.random.uniform(low=1.2, high=2, size=n_res)
    save_mu = mu

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
    test_x = data_prep_tdr_lstm(x_test, n_steps, t_star_test, n_neu, n_res)
    if outputstyle == 'ts':
        np.savez_compressed('Reservoir', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)
        np.save('test_x.npy', test_x)
        np.save('test_y.npy', test_y)
        save = np.stack((error, save_mu, save_nu), axis=1)
        np.savetxt("values.csv", save, delimiter=',', header='error,mu,nu', comments='')
    elif outputstyle in ['mean_sd', 'var', 'var_t']:
        np.savez_compressed('Reservoir', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                            returns_var_test=returns_var_test)
        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)
        np.save('test_x.npy', test_x)
        np.save('test_y.npy', test_y)
        np.save('returns_var_test.npy', returns_var_test)
        save = np.stack((error, save_mu, save_nu), axis=1)
        np.savetxt("values.csv", save, delimiter=',', header='error,mu,nu', comments='')
    return None


def data_prep_tdr_lstm(x, n_steps, t_star, n_neu, n_res):
    x_temp = np.zeros([t_star-n_steps+1, n_neu*n_steps, n_res])
    for i in range(n_res):
        x_temp[:, :, i] = window_stack(x[:, :, i], 1, n_steps)
    return x_temp


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


def gen_u(data, w_in):
    u = np.matmul(w_in, data.transpose())
    return u


def gen_I(u, t_star):
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
    w_out = np.matmul(np.linalg.inv(np.add(np.matmul(x, np.transpose(x)), lamb*np.eye(N=n_neu, M=n_neu))),
                      np.matmul(x, y))
    return w_out


def functional(x, I, nu, mu, p):
    z = (nu*(x+mu*I))/(1+(x+mu*I)**p)
    return z


# Functions maybe not needed
def getoutputshape(outputstyle):
    if outputstyle in ['call', 'put']:
        return 1
    elif outputstyle == 'both':
        return 2
    else:
        print('choose "call", "put" or "both"')

def getinputshape(layer, n_steps, features):
    if layer == 'lstm':
        return [n_steps, features]
    elif layer == 'dense':
        return [features]