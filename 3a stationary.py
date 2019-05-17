from math import sqrt
import pandas as pd
import numpy as np
import utils
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt #,ExponentialSmoothing,  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
def decomp(frame,name,f,mod='Additive'):
    #frame['Date'] = pd.to_datetime(frame['Date'])
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show() # Uncomment to reshow plot, saved as Figure 1.
    return result
def test_stationarity(timeseries):
    
    rolmean = pd.Series(timeseries).rolling(window=60).mean()
    rolstd = pd.Series(timeseries).rolling(window=60).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    #std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("tl.txt", sep='\t')
series=df["Close"]
rolling = series.rolling(window=60)
rolling_mean = rolling.mean()
df.drop("Close",axis=1)
df["Close"]=rolling_mean
dataframe=df.dropna()
print(dataframe)

name='Close'
test_stationarity(series)

result = decomp(dataframe,name,f=1440)
test_stationarity(result.trend)
test_stationarity(result.seasonal)