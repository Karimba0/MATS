import functions as fu
import parameters as par
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd

# From https://github.com/borisbanushev/stockpredictionai
data_x = pd.read_csv('Merge_BKNG.csv', delimiter=',')[['dt', 'open', 'high', 'low', 'close', 'volume', 'RVOL',
                                                       'logreturns', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '20sd',
                                                       'upper_band', 'lower_band', 'ema', 'momentum',  'fft_5', 'fft_10',
                                                       'fft_100', 'fft_1000', 'r_fft_100', 'r_fft_500', 'r_fft_1000',
                                                       'r_fft_1500', 'High', 'Low', 'Adj Close', 'dt_diff']]
data_y = pd.read_csv('Merge_BKNG.csv', delimiter=',')[['var_t', 'var', 'logreturns', 'mean', 'std']]
# choose subset
data_x = data_x.iloc[:-5, ]
data_y = data_y.iloc[5:, ]
# Get training and test data
(train_x, train_y), (test_x, test_y) = fu.get_feature_importance_data(data_x, data_y, par.outputstyle_feature)
regressor = xgb.XGBRegressor(gamma=0.0, n_estimators=500, base_score=500, colsample_bytree=1, learning_rate=0.05)
xgbModel = regressor.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)],
                         verbose=False)
# plot feature importance
fig = plt.figure(figsize=(8, 10))
plt.xticks(rotation='vertical')
plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=test_x.columns)
plt.title('Feature importance for std prediction')
plt.savefig('feature_importance_std.png')
plt.show()
print(data_x.head())

