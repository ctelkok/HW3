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

def decomp(frame,name,f,mod='Additive'):
    
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    
    result.plot()
    plt.show() 
    return result


import warnings
warnings.filterwarnings("ignore")


original = pd.read_csv("tl.txt", sep='\t')

dataframe = original.head(len(original)-1)
dataframe.info()
name='Close'
train=dataframe[0:5038] 
test=dataframe[5038:]
freq = 1440 
series = dataframe[name]
numbers = np.asarray(series,dtype='float')
result = sm.tsa.seasonal_decompose(numbers,freq=1440,model='Additive')

result.plot()
plt.show() 



def estimate_naive(df, seriesname):
     numbers = np.asarray ( df[seriesname] )
     return float( numbers[-1] )
    
naive = estimate_naive (dataframe, name)
print ("Naive estimation:", naive)
dd= np.asarray(train['Close'])
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
rms = sqrt(mean_squared_error(test['Close'], y_hat.naive))
print("Naive rms is",rms)


# Function for Simple Average
def estimate_simple_average(df,seriesname):
    avg = df[seriesname].mean()
    return avg

simpleaverage = estimate_simple_average(dataframe, name)
print("Simple average estimation:", simpleaverage)
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Close'].mean()
rms = sqrt(mean_squared_error(test.Close, y_hat_avg.avg_forecast))
print("Simple avg. rms is",rms)
# Function for Moving Average
def estimate_moving_average(df,seriesname,windowsize):
    avg = df[seriesname].rolling(windowsize).mean().iloc[-1]
    return avg

months = 1440 
movingaverage = estimate_moving_average(dataframe,name, months)
print("Moving average estimation for last ", months, " months: ", movingaverage)
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['Close'].rolling(1440).mean().iloc[-1]
rms = sqrt(mean_squared_error(test.Close, y_hat_avg.moving_avg_forecast))
print("Moving avg. rms is",rms)


   

def estimate_ses(df, seriesname, alpha=0.2):
    numbers = np.asarray(df[seriesname])
    estimate = SimpleExpSmoothing(numbers).fit(smoothing_level=alpha,optimized=False).forecast(1)
    return estimate

alpha = 0.2
ses = estimate_ses(dataframe, name, alpha)[0]
print("Exponential smoothing estimation with alpha =", alpha, ": ", ses)
y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Close'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
rms = sqrt(mean_squared_error(test.Close, y_hat_avg.SES))
print("Ses rms is",rms)
# Trend estimation with Holt
def estimate_holt(df, seriesname, alpha=0.2, slope=0.1):
    numbers = np.asarray(df[seriesname])
    model = Holt(numbers)
    fit = model.fit(alpha,slope)
    estimate = fit.forecast(1)[-1]
    return estimate



alpha = 0.2
slope = 0.1
holt = estimate_holt(dataframe,name,alpha, slope)
print("Holt trend estimation with alpha =", alpha, ", and slope =", slope, ": ", holt)
y_hat_avg = test.copy()

fit1 = Holt(np.asarray(train['Close'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.Close, y_hat_avg.Holt_linear))
print("Holt rms is",rms)
