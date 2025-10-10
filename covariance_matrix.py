import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

np.set_printoptions(legacy='1.25')

#Goals
# Take in a list of securities, and produce a covariance matrix of their returns
class CovarianceMatrix:
    def __init__(self,
                 symbol_list,
                 start = str(date.today() - timedelta(59)),
                 end = str(date.today() - timedelta(1)),
                 interval = '1d',
                 optional_df=None):
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(symbol_list, start, end, interval = interval, multi_level_index=False, ignore_tz=True)
            self.df = df
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
        self.symbol_list = symbol_list
        self.df['Day Count'] = np.arange(1, len(self.df) + 1)
        for symbol in symbol_list:
            self.df[f'{symbol} Return'] = np.log(self.df[f'Close {symbol}']).diff()
        self.df.dropna(inplace = True)
        self.interval = interval
        #Preliminary Calculations
        self.idx = self.df.index.get_loc(self.df.index[0])

    def generate_covariance_matrix(self):
        def Covariance_calc(return_array_1, return_array_2): # Covariance numerator
            SSxy = 0
            n = len(return_array_1)
            index = return_array_1.index.get_loc(return_array_1.index[-1])
            array1_mean = return_array_1.mean()
            array2_mean = return_array_2.mean()
            for return_idx in range(0, index):
                SSxy += (return_array_1.iloc[return_idx] - array1_mean) * (return_array_2.iloc[return_idx] - array2_mean)
            return SSxy / (n - 1)
        def Variance_calc(return_array): # variance numerator 1
            #Works as SSyy also
            SSxx = 0
            n = len(return_array)
            index = return_array.index.get_loc(return_array.index[-1])
            array1_mean = return_array.mean()
            for return_idx in range(0, index):
                SSxx += ((return_array.iloc[return_idx] - array1_mean)**2)
            return SSxx / (n - 1)
        cov_matrix = pd.DataFrame(index=self.symbol_list, columns=self.symbol_list)
        for i in range(len(self.symbol_list)):
            for j in range(len(self.symbol_list)):
                if i == j:
                    cov_matrix.iloc[i,j] = Variance_calc(self.df[f'{self.symbol_list[i]} Return'])
                else:
                    cov_matrix.iloc[i,j] = Covariance_calc(self.df[f'{self.symbol_list[i]} Return'], self.df[f'{self.symbol_list[j]} Return'])
        print("Covariance Matrix of Returns:")
        print("Covariance measures the directional relationship between two assets")
        print("Presented as percentages")
        return cov_matrix * 100
    def generate_correlation_matrix(self):
        def Correlation_calc(return_array_1, return_array_2): # Correlation numerator
            SSxy = 0
            SSxx = 0
            SSyy = 0
            n = len(return_array_1)
            index = return_array_1.index.get_loc(return_array_1.index[-1])
            array1_mean = return_array_1.mean()
            array2_mean = return_array_2.mean()
            for return_idx in range(0, index):
                SSxy += (return_array_1.iloc[return_idx] - array1_mean) * (return_array_2.iloc[return_idx] - array2_mean)
                SSxx += ((return_array_1.iloc[return_idx] - array1_mean)**2)
                SSyy += ((return_array_2.iloc[return_idx] - array2_mean)**2)
            return SSxy / math.sqrt(SSxx) * math.sqrt(SSyy)
        corr_matrix = pd.DataFrame(index=self.symbol_list, columns=self.symbol_list)
        for i in range(len(self.symbol_list)):
            for j in range(len(self.symbol_list)):
                if i == j:
                    corr_matrix.iloc[i,j] = 1
                else:
                    corr_matrix.iloc[i,j] = Correlation_calc(self.df[f'{self.symbol_list[i]} Return'], self.df[f'{self.symbol_list[j]} Return'])
        print("Correlation Matrix of Returns:")
        print("Correlation measures the strength of relationship between two assets")
        print("Presented as percentages")
        return corr_matrix * 100
    def pairs_closest_to_0(self, n=5):
        pass