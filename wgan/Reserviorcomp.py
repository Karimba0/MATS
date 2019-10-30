import pandas as pd
import functions as fu
import parameters as par


# data
# data = pd.read_csv('merge_AAPL.csv', delimiter=',')[['RVOL', 'Adj Close', 'logreturns']].to_numpy()
data = pd.read_csv(par.data_name, delimiter=',')[['open', 'high', 'low', 'close', 'volume', 'RVOL', 'ma7', 'ma21',
                                                  '26ema', '12ema', 'MACD', '20sd', 'upper_band', 'lower_band', 'ema',
                                                  'High', 'Low', 'Adj Close', 'dt_diff', 'logreturns']].to_numpy()
data_c = pd.read_csv('merge_AAPL.csv', delimiter=',')[['close']].to_numpy()
data_y = pd.read_csv('merge_AAPL.csv', delimiter=',')[['logreturns']].to_numpy()
dates = pd.read_csv('merge_AAPL.csv', delimiter=',')[['dt']]

# First prediction index test set
n_split = fu.get_datesplit(par.date, dates)
# compute reservoir
fu.compute_reservoir_wgan(par.hor, n_split, par.ts_out_length, par.n, par.n_neu, par.n_i, par.k_i,
                                   par.h, par.n_res, par.n_input_layers, data, data_y)
