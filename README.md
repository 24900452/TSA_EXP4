# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 6.11.2026
 # Name : Jothi Ganesh P
 # REG no:212224240065


### AIM:
To implement ARMA model in python.
### dataset:
nifty_it_index

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


data = pd.read_csv('/content/nifty_it_index.csv', parse_dates=['Date'], index_col='Date')
data = data.sort_index()

N = 1000
plt.rcParams['figure.figsize'] = [12, 6]


plt.plot(data['Close'])
plt.title('Original Data - Nifty IT Close')
plt.show()
print('ADF p-value (Close):', adfuller(data['Close'])[1])


X = np.log(data['Close']).diff().dropna()
plt.plot(X)
plt.title('Stationary Data - Nifty IT Daily Log Returns')
plt.show()
print('ADF p-value (log returns):', adfuller(X)[1])

plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()


arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
print(arma11_model.summary())
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']


ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()


arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
print(arma22_model.summary())
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']


ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="993" height="528" alt="download" src="https://github.com/user-attachments/assets/1b562ed8-2ef1-4a59-ac4a-fa18be78e06e" />



Partial Autocorrelation
<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/41df696b-6070-43e4-b7bc-19f2ffaf17a0" />

Autocorrelation

<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/2b7ce09d-3ea7-404b-af4e-97ee2c16a518" />


SIMULATED ARMA(2,2) PROCESS:
<img width="993" height="528" alt="download" src="https://github.com/user-attachments/assets/d0a114c2-3747-4441-a1bc-68ad191f03be" />

Partial Autocorrelation

<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/da20e6a1-a8c0-445f-b39e-39bb0f15aaeb" />


Autocorrelation
<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/15882987-c39d-4b31-aa3d-91d451e68b66" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
