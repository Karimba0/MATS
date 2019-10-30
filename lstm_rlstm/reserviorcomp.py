import functions as fu
import pandas as pd
import parameters as par


data_name = 'merge_AAPL.csv'
# data
data = pd.read_csv(par.data_name, delimiter=',')[['volume', 'RVOL', 'MACD', '20sd', 'High', 'Low', 'Adj Close',
                                                  'dt_diff', 'r_fft_100', 'r_fft_500', 'logreturns']]
data_c = pd.read_csv('merge_AAPL.csv', delimiter=',')[['close']].to_numpy()
data_y = pd.read_csv('merge_AAPL.csv', delimiter=',')[['logreturns']]
dates = pd.read_csv(data_name, delimiter=',')[['dt']]
# split date
n_split = fu.get_datesplit(par.date, dates)
t_train = n_split
t_test = data.shape[0]-par.n_steps-n_split
# calculate reservoir
fu.compute_reservoir_lstm(data, data_y, par.outputstyle, n_split, par.rolling_var, par.n_input_layers,
                          par.ts_out_length, par.hor, par.alpha, par.layer, par.reservoir, par.dof, par.k_i,
                          par.n_i, par.n, par.n_res, par.h, par.n_neu)

