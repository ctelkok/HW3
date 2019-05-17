import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt 
from statsmodels.tsa.stattools import adfuller

def decomp(frame,name,f,mod='Additive'):
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    
    result.plot()
    plt.show() 
    return result

 
def test_stationarity(timeseries):
    
    rolmean = pd.Series(timeseries).rolling(window=2).mean()
    rolstd = pd.Series(timeseries).rolling(window=2).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
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


# Load data
df = pd.read_csv("tl.txt", sep='\t')
seriesname = 'Close' 
df['Daytime'] = pd.to_datetime(df['Day'] +' ' + df['Time'])
df = df.set_index('Daytime')
df = df.drop(['Day','Time'],axis=1)
dataframe=df.between_time("15:57","15:58")
name='Close'


indexNames = dataframe[ dataframe['Volume'] == 0 ].index

dataframe.drop(indexNames , inplace=True)

train=dataframe[0:7] 
test=dataframe[7:]
freq = 1 
series = dataframe[name]
numbers = np.asarray(series,dtype='float')
result = sm.tsa.seasonal_decompose(numbers,freq=2,model='Additive')
import warnings
warnings.filterwarnings("ignore")
test_stationarity(series)

result = decomp(dataframe,name,f=1)
test_stationarity(result.trend)
test_stationarity(result.seasonal)
