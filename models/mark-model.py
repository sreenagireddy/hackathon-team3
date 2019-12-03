import pandas as pd
from matplotlib import pyplot
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tas.arima_model import ARIMA
import statistics

series = pd.read_csv('filename.csv',
					index_col=0,
					parse_dates=True,
					squeeze=True)

print(series)
