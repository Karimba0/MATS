import numpy as np
import pandas as pd
import functions as fu
import os


vix = pd.read_csv('VIX.csv', delimiter=',')[['dt', 'High', 'Low', 'Adj Close']]
vix['dt'] = pd.to_datetime(vix['dt'], errors='coerce')
vix['dt'] = vix['dt'].dt.date
os.chdir('C:\Python\Data\Stock_Data_test')

# Hyperparameters for RVOL
alpha = 1.34
n = 5
# trading days
trading_days = 252

for filename in os.listdir(os.getcwd()):
    print(filename)
    df = pd.read_csv(filename, delimiter=',', index_col=[0])
    df = df.reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df['dt_diff'] = df['dt'].diff(periods=-1).dt.days
    V = fu.calc_rvol(df, alpha, n, trading_days)
    V = np.append(0, V)
    df['RVOL'] = V
    df['logreturns'] = np.append(0, np.diff(np.log(df['close'].to_numpy())))
    df = fu.get_data_ind(df, include_indicators=True, include_fft=True)
    print(df.head())
    df = df.dropna()
    df['dt'] = df['dt'].dt.date
    merge = pd.merge(df, vix, on='dt')
    merge.to_csv('Merge_' + filename, sep=',', index=False)
