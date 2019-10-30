# MATS
 Trading Strategies based on the gap between Implied and Realized Volatility: A machine learning approach

### Packages needed in Python:
* python 3.6
* tensorflow 2.0
* tqdm
* pandas
* numpy
* sklearn
* joblib
* keras-rectified-adam
* seaborn
* scipy
* xgboost
* matplotlib

### Walkthrough LSTM code:
1. Set Parameters in parameters.py 
2. Run init_LSTM to generate the network and save it. (If reservoir is set to True we first need to run reservoircomp.py to get the Reservoir data)(only produces one network)
3. Continue training models
	1. If rolling is False we run single_lstm.py (epochs and batch_size are specified in the File itself)
	2. If rolling is True we run rolling_lstm.py (epochs and batch_size are specified in the File itself). In the first run of rolling_lstm.py we generate the different networks for the rolling forecast. After that we set continue to True and train the networks
4. Models will be saved and if they are not fitted well enough we can repeat step 3.1 or 3.2 .
5. Evaluate models
	1. If we want to evlauate the single model run evaluate_single_lstm. 
	2. If we want to evaluate the rolling_lstm model we set evaluate and continue to True.
6. To do the dynamic quantile test and the conditional covergage test specify the safe path of the cond_cov.csv(for single_lstm) or cond_cov_r(for rolling_lstm) in backtest_cond_cov.py.

### Walkthrough WGAN code:
1. Set Parameters in parameters.py 
2. Run init_wgan.py to generate the network. (If reservoir is set to True we first need to run reservoircomp.py to get the Reservoir data)
3. Run continue_training_wgan.py to train WGAN further. (epochs and batch_size are specified in the File itself)
4. Run calculate_VaR.py to estimate VaR for the specified test set.
5. To do the dynamic quantile test and the conditional covergage test specify the safe path of cond_cov.csv(output of calculate_VaR.py) and set rolling to False

### EVT:
Packages needed:
evir
rugarch
ggplot2
xts

Comments in code describe what needs to be set before running.
Then run the whole evt.r program.

### Comments and references
* plots for fourier and feature importance from https://github.com/borisbanushev/stockpredictionai
* parts of the code for WGAN from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2
* feature importance and the the creation of the data files are in the wgan directory
* Values for mean std and VAR have fixed rolling_var when generating the data files
* For networks rolling_var can be reset because it gets calculated again
