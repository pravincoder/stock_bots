import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import warnings 
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller

# Function to test for stationarity
def adf_test(series, title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    

# Load the data
data = pd.read_csv('./NIFTY 50-11-06-2023-to-11-06-2024.csv')

# Clean the column names
data.columns = data.columns.str.strip()

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date']) 
data.set_index('Date', inplace=True) 

# Select the features for forecasting
features = data[['Open', 'High', 'Low', 'Close','Shares Traded','Turnover (₹ Cr)']]

# Check stationarity for each feature
for column in features.columns:
    adf_test(features[column], title=column)

# Make Data Stationary
features_diff = features.diff().dropna()

# Check stationarity again after differencing
for column in features_diff.columns:
    adf_test(features_diff[column], title=f'{column} Differenced')

# Load the VAR model on the differenced data
model = VAR(features_diff)
lag_order = model.select_order(maxlags=20)
#print(lag_order.summary())

# Fit the VAR model
model_fitted = model.fit(lag_order.aic)
#print(model_fitted.summary())

# Extract residuals
residuals = model_fitted.resid

# Save the model
model_fitted.save('var_model.pkl')