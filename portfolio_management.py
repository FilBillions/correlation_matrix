import yfinance as yf
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
sys.path.append(".")
import csv

from datetime import date, timedelta
from sampler import Sampler

np.set_printoptions(legacy='1.25')
def convert_portfolio_to_symbol_list(return_symbols=False, return_weights=False):
    #look at portfolio weights file and convert to a list of the symbols only
    with open('portfolio_weights.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        symbol_list = []
        weight_dict = {}
        weight_list = []
        for row in csv_reader:
            symbol_list.append(row['Ticker'])
            weight_list.append(row['Weight'])
    if return_symbols:
        return symbol_list
    if return_weights:
        weight_dict = dict(zip(symbol_list, weight_list))
        return weight_dict

class PortfolioManagement:
    def __init__(self,
                 symbol_list, 
                 start = str(date.today() - timedelta(59)),
                 end = str(date.today() - timedelta(1)),
                 interval = '1d',
                 optional_df=None,
                 annualize=False):
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
        self.weight_dict = convert_portfolio_to_symbol_list(return_weights=True)
        self.df['Day Count'] = np.arange(1, len(self.df) + 1)
        if annualize and interval == '1d':
            from daily_to_yearly_conversion import Converter
            converter = Converter(self.df, self.symbol_list)
            self.df = converter.yearly_df
        else:
            for symbol in self.symbol_list:
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
        for symbol in self.symbol_list:
            returns_array = self.df[f'{symbol} Return']
            n = len(returns_array)
            mean_return = returns_array.mean()
            variance_return = 0
            for return_idx in range(0, n):
                variance_return += (returns_array.iloc[return_idx] - mean_return)**2
            variance_return = variance_return / (n - 1)
            self.mean_dict[symbol] = mean_return
            self.variance_dict[symbol] = variance_return

    def generate_covariance_matrix(self, print_on=True, return_on=False):
        def Covariance_calc(return_array_1, return_array_2): # Covariance numerator
            SSxy = 0
            n = len(return_array_1)
            index = return_array_1.index.get_loc(return_array_1.index[-1])
            array1_mean = return_array_1.mean()
            array2_mean = return_array_2.mean()
            for return_idx in range(0, index):
                SSxy += (return_array_1.iloc[return_idx] - array1_mean) * (return_array_2.iloc[return_idx] - array2_mean)
            cov = SSxy / (n - 1)
            return cov
        def Variance_calc(return_array): # variance numerator 1
            #Works as SSyy also
            SSxx = 0
            n = len(return_array)
            index = return_array.index.get_loc(return_array.index[-1])
            array1_mean = return_array.mean()
            for return_idx in range(0, index):
                SSxx += ((return_array.iloc[return_idx] - array1_mean)**2)
            var_x = SSxx / (n - 1)
            return var_x
        cov_matrix = pd.DataFrame(index=self.symbol_list, columns=self.symbol_list)
        for i in range(len(self.symbol_list)):
            for j in range(len(self.symbol_list)):
                if i == j:
                    cov_matrix.iloc[i,j] = Variance_calc(self.df[f'{self.symbol_list[i]} Return'])
                else:
                    cov_matrix.iloc[i,j] = Covariance_calc(self.df[f'{self.symbol_list[i]} Return'], self.df[f'{self.symbol_list[j]} Return'])
        if print_on:
            fig, ax = plt.subplots(figsize=(10, 8))
            rdgn = sns.diverging_palette(h_neg=355, h_pos=255, s=100, sep=1, l=50, as_cmap=True)
            sns.heatmap(cov_matrix, annot=True, fmt=".3f", cmap=rdgn, vmin=-1, center= 0, vmax=1, linewidths=1.3, linecolor='black', cbar=True, ax=ax)
            plt.title('Covariance Matrix')
            print("Covariance measures the directional relationship between two assets")
        if return_on:
            return cov_matrix
    
    def generate_correlation_matrix(self, print_on=True, return_on=False):
        def Correlation_calc(return_array_1, return_array_2): # Correlation numerator
            SSxy = 0
            SSxx = 0
            SSyy= 0
            n = len(return_array_1)
            index = return_array_1.index.get_loc(return_array_1.index[-1])
            array1_mean = return_array_1.mean()
            array2_mean = return_array_2.mean()
            for return_idx in range(0, index):
                SSxy += (return_array_1.iloc[return_idx] - array1_mean) * (return_array_2.iloc[return_idx] - array2_mean)
                SSxx += ((return_array_1.iloc[return_idx] - array1_mean)**2)
                SSyy += ((return_array_2.iloc[return_idx] - array2_mean)**2)
            cov = SSxy / (n - 1)
            var_xx = SSxx / (n - 1)
            var_yy = SSyy / (n - 1)
            return cov / (math.sqrt(var_xx) * math.sqrt(var_yy))
        corr_matrix = pd.DataFrame(index=self.symbol_list, columns=self.symbol_list)
        for i in range(len(self.symbol_list)):
            for j in range(len(self.symbol_list)):
                if i == j:
                    corr_matrix.iloc[i,j] = 1
                else:
                    corr_matrix.iloc[i,j] = Correlation_calc(self.df[f'{self.symbol_list[i]} Return'], self.df[f'{self.symbol_list[j]} Return'])
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
        
    def calculate_statistics(self, return_portfolio_only=False):
        correlation_matrix = self.generate_correlation_matrix(print_on=False, return_on=True)
        covariance_matrix = self.generate_covariance_matrix(print_on=False, return_on=True)
        market_symbol = self.symbol_list[0]
        beta_dict = {}
        expected_return = {}
        sharpe_ratio_dict = {}
        m2_alpha_dict = {}
        jensens_alpha_dict = {}
        tracking_error_dict = {}
        information_ratio_dict = {}
        expected_market_return = self.mean_dict[market_symbol]
        annual_risk_free_rate = .02
#Beta Calculation
        for symbol in self.symbol_list[0:]:
            beta = (correlation_matrix.loc[symbol, market_symbol] *
                    (math.sqrt(self.variance_dict[symbol]) / math.sqrt(self.variance_dict[market_symbol])))
            beta_dict[symbol] = beta
        statistic_df = pd.DataFrame.from_dict(beta_dict, orient='index', columns=['Beta'])
#Weights
        for symbol in self.symbol_list[0:]:
            if symbol in self.weight_dict:
                statistic_df.loc[symbol, 'Actual Weight'] = float(self.weight_dict[symbol])
            else:
                statistic_df.loc[symbol, 'Actual Weight'] = 0.0
# tracking error calculation
        for symbol in self.symbol_list[0:]:
            tracking_error = 0
            n = len(self.df)
            index = self.df.index.get_loc(self.df.index[-1])
            for return_idx in range(0, index):
                tracking_error += (self.df[f'{symbol} Return'].iloc[return_idx] - self.df[f'{market_symbol} Return'].iloc[return_idx])**2
            tracking_error_var = math.sqrt(tracking_error / (n - 1))
            tracking_error_dict[symbol] = tracking_error_var
#expected return calculation using CAPM
        for symbol in self.symbol_list[0:]:
            expected_return[symbol] = annual_risk_free_rate + beta_dict[symbol] * (expected_market_return - annual_risk_free_rate)
        statistic_df['Expected Return'] = pd.Series(expected_return)
#variance
        statistic_df['Variance'] = pd.Series(self.variance_dict)
#sharpe ratio
        for symbol in self.symbol_list[0:]:
            sharpe_ratio = (expected_return[symbol] - annual_risk_free_rate) / math.sqrt(self.variance_dict[symbol])
            sharpe_ratio_dict[symbol] = sharpe_ratio
        statistic_df['Sharpe Ratio'] = pd.DataFrame.from_dict(sharpe_ratio_dict, orient='index', columns=['Sharpe Ratio'])
#m2
        for symbol in self.symbol_list[0:]:
            m2 = (expected_return[symbol] - annual_risk_free_rate) * (math.sqrt(self.variance_dict[market_symbol]) / math.sqrt(self.variance_dict[symbol])) + annual_risk_free_rate
            m2_alpha_dict[symbol] = m2 - expected_market_return
        statistic_df['M2 Alpha'] = pd.DataFrame.from_dict(m2_alpha_dict, orient='index', columns=['M2 Alpha'])
# treynor ratio
# use actual returns for treynor ratio
        for symbol in self.symbol_list[0:]:
            treynor_ratio = (self.mean_dict[symbol] - annual_risk_free_rate) / beta_dict[symbol]
            statistic_df.loc[symbol, 'Treynor Ratio'] = treynor_ratio
#jensen's alpha -> based on the mean return
# jensens alpha has to use actual returns, it cannot use expected returns
# do not perform calculation for market symbol
        for symbol in self.symbol_list[0:]:
            if symbol == market_symbol:
                jensens_alpha_dict[symbol] = 0
            else:
                jensens_alpha = (self.mean_dict[symbol] - (annual_risk_free_rate + beta_dict[symbol] * (expected_market_return - annual_risk_free_rate)))
                jensens_alpha_dict[symbol] = jensens_alpha
        statistic_df['Jensen\'s Alpha'] = pd.DataFrame.from_dict(jensens_alpha_dict, orient='index', columns=['Jensen\'s Alpha'])
#information ratio
        for symbol in self.symbol_list[0:]:
            if symbol == market_symbol:
                information_ratio_dict[symbol] = 0
            else:
                information_ratio = jensens_alpha_dict[symbol] / tracking_error_dict[symbol]
                information_ratio_dict[symbol] = information_ratio
        statistic_df['Information Ratio'] = pd.DataFrame.from_dict(information_ratio_dict, orient='index', columns=['Information Ratio'])
#Weight Difference
        for symbol in self.symbol_list[0:]:
            statistic_df.loc[symbol, 'Weight Difference'] = statistic_df.loc[symbol, 'Actual Weight'] - statistic_df.loc[symbol, 'Information Ratio']

# -------------------------------
# -------------------------------
#Start portfolio

        portfolio_row = pd.DataFrame(columns=statistic_df.columns)
#portfolio expected return
# 1 - Actual Weight of portfolio should be ivnested in the risk free rate!!!
# portfolio weight is the sum of all the securities in the portfolio
        portfolio_row.loc['Portfolio', 'Actual Weight'] = statistic_df['Actual Weight'].sum()
        for symbol in self.symbol_list[0:]:
            if symbol == self.symbol_list[0]:
                portfolio_expected_return = statistic_df.loc[symbol, 'Expected Return'] * statistic_df.loc[symbol, 'Actual Weight']
            else:
                portfolio_expected_return += statistic_df.loc[symbol, 'Expected Return'] * statistic_df.loc[symbol, 'Actual Weight']
        portfolio_expected_return += (1 - portfolio_row.loc['Portfolio', 'Actual Weight']) * annual_risk_free_rate
        portfolio_row.loc['Portfolio', 'Expected Return'] = portfolio_expected_return
#portfolio variance
#portfolio variance is calculated = W**2 * Var + W*W*COV
# we need to throw Cash in the variance here too
        sum_of_covariance_terms = 0
        for i, symbol_1 in enumerate(self.symbol_list[0:]):
            for j, symbol_2 in enumerate(self.symbol_list[0:]):
                weight_1 = statistic_df.loc[symbol_1, 'Actual Weight']
                weight_2 = statistic_df.loc[symbol_2, 'Actual Weight']
                covariance = covariance_matrix.loc[symbol_1, symbol_2]
                if i != j:
                    sum_of_covariance_terms += weight_1 * weight_2 * covariance
        sum_of_variance_terms = 0
        for symbol in self.symbol_list[0:]:
            weight = statistic_df.loc[symbol, 'Actual Weight']
            variance = statistic_df.loc[symbol, 'Variance']
            sum_of_variance_terms += (weight**2) * variance
        portfolio_variance = sum_of_variance_terms + sum_of_covariance_terms
        portfolio_row.loc['Portfolio', 'Variance'] = portfolio_variance
#portfolio beta -> maybe look at this later
        portfolio_beta = 0
        for symbol in self.symbol_list[0:]:
            portfolio_beta += statistic_df.loc[symbol, 'Beta'] * statistic_df.loc[symbol, 'Actual Weight']
        portfolio_row.loc['Portfolio', 'Beta'] = portfolio_beta
#portfolio sharpe ratio
        portfolio_sharpe_ratio = (portfolio_expected_return - annual_risk_free_rate) / math.sqrt(portfolio_variance)
        portfolio_row.loc['Portfolio', 'Sharpe Ratio'] = portfolio_sharpe_ratio
#portfolio m2 alpha
        portfolio_m2 = (portfolio_expected_return - annual_risk_free_rate) * (math.sqrt(self.variance_dict[market_symbol]) / math.sqrt(portfolio_variance)) + annual_risk_free_rate
        portfolio_row.loc['Portfolio', 'M2 Alpha'] = portfolio_m2 - expected_market_return
#portfolio treynor ratio
        portfolio_treynor_ratio = (portfolio_expected_return - annual_risk_free_rate) / portfolio_beta
        portfolio_row.loc['Portfolio', 'Treynor Ratio'] = portfolio_treynor_ratio
#portfolio jensens alpha
        portfolio_jensens_alpha = (portfolio_expected_return - (annual_risk_free_rate + 1 * (expected_market_return - annual_risk_free_rate)))
        portfolio_row.loc['Portfolio', 'Jensen\'s Alpha'] = portfolio_jensens_alpha
#portfolio weight is the sum of all the securities in the portfolio
        portfolio_row.loc['Portfolio', 'Actual Weight'] = statistic_df['Actual Weight'].sum()
        portfolio_row = portfolio_row.dropna(axis=1, how='any')
        statistic_df = pd.concat([statistic_df, portfolio_row])

# -------------------------------
# -------------------------------
#Start optimized portfolio

        optimal_portfolio_row = pd.DataFrame(columns=statistic_df.columns)
        optimal_portfolio_weight = 0
#portfolio expected return
# 1 - Information Ratio of portfolio should be ivnested in the risk free rate!!!
# portfolio weight is the sum of all the information ratios in the portfolio
# if information ratio is negative, do not include in portfolio
        for symbol in self.symbol_list[0:]:
            if statistic_df.loc[symbol, 'Information Ratio'] > 0:
                optimal_portfolio_weight += statistic_df.loc[symbol, 'Information Ratio']
        optimal_portfolio_row.loc['Optimal Portfolio', 'Actual Weight'] = optimal_portfolio_weight
        for symbol in self.symbol_list[0:]:
            if symbol == self.symbol_list[0]:
                optimal_portfolio_expected_return = statistic_df.loc[symbol, 'Expected Return'] * statistic_df.loc[symbol, 'Information Ratio']
            else:
                optimal_portfolio_expected_return += statistic_df.loc[symbol, 'Expected Return'] * statistic_df.loc[symbol, 'Information Ratio']
        optimal_portfolio_expected_return += (1 - optimal_portfolio_row.loc['Optimal Portfolio', 'Actual Weight']) * annual_risk_free_rate
        optimal_portfolio_row.loc['Optimal Portfolio', 'Expected Return'] = optimal_portfolio_expected_return
    #portfolio variance
    #portfolio variance is calculated = W**2 * Var + W*W*COV
        sum_of_covariance_terms = 0
        for i, symbol_1 in enumerate(self.symbol_list[0:]):
            for j, symbol_2 in enumerate(self.symbol_list[0:]):
                weight_1 = statistic_df.loc[symbol_1, 'Information Ratio']
                weight_2 = statistic_df.loc[symbol_2, 'Information Ratio']
                covariance = covariance_matrix.loc[symbol_1, symbol_2]
                if i != j:
                    sum_of_covariance_terms += weight_1 * weight_2 * covariance
        sum_of_variance_terms = 0
        for symbol in self.symbol_list[0:]:
            weight = statistic_df.loc[symbol, 'Information Ratio']
            variance = statistic_df.loc[symbol, 'Variance']
            sum_of_variance_terms += (weight**2) * variance
        optimal_portfolio_variance = sum_of_variance_terms + sum_of_covariance_terms
        optimal_portfolio_row.loc['Optimal Portfolio', 'Variance'] = optimal_portfolio_variance
        #portfolio beta -> maybe look at this later
        optimal_portfolio_beta = 0
        for symbol in self.symbol_list[0:]:
            optimal_portfolio_beta += statistic_df.loc[symbol, 'Beta'] * statistic_df.loc[symbol, 'Information Ratio']
        optimal_portfolio_row.loc['Optimal Portfolio', 'Beta'] = optimal_portfolio_beta
        #portfolio sharpe ratio
        optimal_portfolio_sharpe_ratio = (optimal_portfolio_expected_return - annual_risk_free_rate) / math.sqrt(optimal_portfolio_variance)
        optimal_portfolio_row.loc['Optimal Portfolio', 'Sharpe Ratio'] = optimal_portfolio_sharpe_ratio
        #portfolio m2 alpha
        optimal_portfolio_m2 = (optimal_portfolio_expected_return - annual_risk_free_rate) * (math.sqrt(self.variance_dict[market_symbol]) / math.sqrt(optimal_portfolio_variance)) + annual_risk_free_rate
        optimal_portfolio_row.loc['Optimal Portfolio', 'M2 Alpha'] = optimal_portfolio_m2 - expected_market_return
        #portfolio treynor ratio
        optimal_portfolio_treynor_ratio = (optimal_portfolio_expected_return - annual_risk_free_rate) / optimal_portfolio_beta
        optimal_portfolio_row.loc['Optimal Portfolio', 'Treynor Ratio'] = optimal_portfolio_treynor_ratio
        #portfolio jensens alpha
        optimal_portfolio_jensens_alpha = (optimal_portfolio_expected_return - (annual_risk_free_rate + 1 * (expected_market_return - annual_risk_free_rate)))
        optimal_portfolio_row.loc['Optimal Portfolio', 'Jensen\'s Alpha'] = optimal_portfolio_jensens_alpha
        optimal_portfolio_row = optimal_portfolio_row.dropna(axis=1, how='all')
        statistic_df = pd.concat([statistic_df, optimal_portfolio_row])
        if return_portfolio_only:
            print('Portfolio Comparison to Market:')
            if optimal_portfolio_sharpe_ratio > sharpe_ratio_dict[market_symbol]:
                print(f"Portfolio has a higher Sharpe Ratio than the Market")
                print(f"Therefore, Portfolio is receiving more return per unit of risk taken than market")
            else:
                print(f"Market has a higher Sharpe Ratio than the Portfolio")
                print(f"Therefore, Market is receiving more return per unit of risk taken than portfolio")
                print("Portfolio may need to be rebalanced to improve performance")
            statistic_df = statistic_df.loc[[market_symbol, 'Portfolio', 'Optimal Portfolio']]
            return statistic_df.drop(columns=['Weight Difference', 'Information Ratio'])
        
        return statistic_df
    def return_df(self):
        #show all rows
        pd.set_option('display.max_rows', None)
        return self.df