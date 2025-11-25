import yfinance as yf
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from datetime import date, timedelta
from sampler import Sampler

np.set_printoptions(legacy='1.25')


#Goals
# Take in a list of securities, and produce a covariance matrix of their returns
class PortfolioManagement:
    def __init__(self,
                 symbol_list,
                 start = str(date.today() - timedelta(59)),
                 end = str(date.today() - timedelta(1)),
                 interval = '1d',
                 optional_df=None):
        market_symbol = 'SPY'
        if market_symbol not in symbol_list:
            symbol_list.insert(0, market_symbol)
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
        #insert sampler here
        sampler = Sampler(self.df)
        self.df = sampler.sampled_df
        self.interval = interval
        #Preliminary Calculations
        self.idx = self.df.index.get_loc(self.df.index[0])
        self.mean_dict = {}
        self.variance_dict = {}
        for symbol in symbol_list:
            returns_array = self.df[f'{symbol} Return']
            n = len(returns_array)
            mean_return = returns_array.mean()
            variance_return = 0
            for return_idx in range(0, n):
                variance_return += (returns_array.iloc[return_idx] - mean_return)**2
            variance_return = variance_return / (n - 1)
            self.mean_dict[symbol] = mean_return
            self.variance_dict[symbol] = variance_return
        print(self.variance_dict)

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
    
    def generate_correlation_matrix(self, print_on=True, return_on=False):
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
        #apply color gradient to dataframe
        #vmin should be the smallest value in the matrix excluding the diagonal
        min_val = corr_matrix.min().min()
        #set heatmap dimensions
        corr_matrix = corr_matrix.astype(float)
        if print_on:
            fig, ax = plt.subplots(figsize=(10, 8))
            rdgn = sns.diverging_palette(h_neg=355, h_pos=255, s=100, sep=1, l=50, as_cmap=True)
            sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap=rdgn, vmin=-1, center= 0, vmax=1, linewidths=1.3, linecolor='black', cbar=True, ax=ax)
            plt.title('Correlation Matrix')
            print("Correlation measures the strength of relationship between two assets")
            print("Closer to 1 means strong positive correlation, closer to -1 means strong negative correlation")
            print("A value of 0 means no correlation")
            plt.show()
        if return_on:
            return corr_matrix
    def calculate_beta(self):
        correlation_matrix = self.generate_correlation_matrix(print_on=False, return_on=True)
        market_symbol = self.symbol_list[0]
        beta_dict = {}
        # for every symbol except the market symbol
        for symbol in self.symbol_list[1:]:
            beta = (correlation_matrix.loc[symbol, market_symbol] *
                    (math.sqrt(self.variance_dict[market_symbol]) / math.sqrt(self.variance_dict[symbol])))
            beta_dict[symbol] = beta
        beta_df = pd.DataFrame.from_dict(beta_dict, orient='index', columns=['Beta'])
        return beta_df

