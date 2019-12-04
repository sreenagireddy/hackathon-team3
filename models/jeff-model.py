#python code
from fbprophet import Prophet
import numpy as np
import pandas as pd
#I created a csv file with closed refi applications with columns: 'y' with $ amounts, and 'ds' with closing dates.
#make sure date format is yyyy-mm-dd
#read csv
sales_df = pd.read_csv('sample.csv')
#convert data to numeric data type
sales_df['y'] = pd.to_numeric(sales_df['y'].str.replace(',',''))
# convert date to date data type
sales_df['ds'] = sales_df['ds'].astype('datetime64[ns]')

# group $ amounts by day
sales_df = sales_df.groupby(['ds'])[['y']].agg('sum')
#backup y column
sales_df['y_orig'] = sales_df['y']
# log-transform y - no idea why.
sales_df['y'] = np.log(sales_df['y'])

# fix column names for the next library - prophet
sales_df['ds'] = sales_df.index
sales_df.index.name = 'test'

#split data into training vs valid date ranges
training_df = sales_df['2016-01-01':'2019-06-30']
valid_df = sales_df['2019-07-01':'2019-10-31']

#Trying out this prophet library
model = Prophet() #instantiate Prophet
model.fit(training_df); #fit the model with your training dataframe
# forecast # days out
future_data = model.make_future_dataframe(periods=130)
# future_data = model.make_future_dataframe(periods=18, freq='w')
forecast_data = model.predict(future_data)
# model.plot(forecast_data).show()
# because we forecasted with log-transformed data, we're reversing that here
forecast_data_orig = forecast_data 
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
training_df['y_log']=training_df['y'] #copy the log-transformed data to another column
training_df['y']=training_df['y_orig'] #copy the original data to 'y'
valid_df['y_log']=valid_df['y'] #copy the log-transformed data to another column
valid_df['y']=valid_df['y_orig'] #copy the original data to 'y'
# plotting the prediction
# model.plot(forecast_data_orig).show()

# set the index correct of the forecast to match our date index
forecast_data_orig.index = forecast_data_orig['ds']

# plot the graph comparing all 3 data sets
import matplotlib.pyplot as plt
plt.plot(training_df['y'], label='Training')
plt.plot(valid_df['2019-07-01':'2019-10-31']['y'], label='Valid')
plt.plot(forecast_data_orig['2019-07-01':'2019-10-31']['yhat'], label='Prediction')
plt.show()

#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error
#doesn't currently work
rms = sqrt(mean_squared_error(valid_df['2019-07-01':'2019-10-31']['y'],forecast_data_orig['2019-07-01':'2019-10-31']['yhat']))
print(rms)
