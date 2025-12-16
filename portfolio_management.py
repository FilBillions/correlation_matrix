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

class PortfolioManagement:
    def __init__(self,
                 portfolio_dictionary,
                 market_symbol = 'SPY',
                 start = str(date.today() - timedelta(59)),
                 end = str(date.today() - timedelta(1)),
                 interval = '1d',
                 optional_df=None,
                 annualize=False):
        symbol_list = list(portfolio_dictionary.keys())
        self.market_symbol = market_symbol.upper()
        if self.market_symbol not in symbol_list:
            symbol_list.insert(0, self.market_symbol)
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(symbol_list, start, end, interval = interval, multi_level_index=False, ignore_tz=True)
            self.df = df
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
        self.symbol_list = symbol_list
        for symbol in self.symbol_list:
            self.df=self.df.drop(columns=[f'Volume {symbol}'])
            self.df=self.df.drop(columns=[f'High {symbol}'])
            self.df=self.df.drop(columns=[f'Low {symbol}'])
            self.df=self.df.drop(columns=[f'Open {symbol}'])
        self.weight_dict = portfolio_dictionary
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
#Preliminary Calculation
        self.annual_risk_free_rate = .02
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
        self.beta_dict = {}
        self.expected_return = {}
        self.sharpe_ratio_dict = {}
        self.m2_alpha_dict = {}
        self.jensens_alpha_dict = {}
        self.nonsystematic_variance_dict = {}
        self.information_ratio_dict = {}
        self.expected_market_return = self.mean_dict[self.market_symbol]
        self.check1_dict = {}
        #this is the weight of the non-market security within only the portfolio of non-market securities
        self.non_market_weightings_only_dict = {}
        self.optimal_portfolio_weighting_dict = {}
#Beta Calculation
        for symbol in self.symbol_list[0:]:
            beta = (correlation_matrix.loc[symbol, self.market_symbol] *
                    (math.sqrt(self.variance_dict[symbol]) / math.sqrt(self.variance_dict[self.market_symbol])))
            self.beta_dict[symbol] = beta
        statistic_df = pd.DataFrame.from_dict(self.beta_dict, orient='index', columns=['Beta'])
#Weights
        for symbol in self.symbol_list[0:]:
            if symbol in self.weight_dict:
                statistic_df.loc[symbol, 'Actual Weight'] = float(self.weight_dict[symbol])
            else:
                statistic_df.loc[symbol, 'Actual Weight'] = 0.0
#expected return calculation using CAPM
        for symbol in self.symbol_list[0:]:
            self.expected_return[symbol] = self.annual_risk_free_rate + self.beta_dict[symbol] * (self.expected_market_return - self.annual_risk_free_rate)
        statistic_df['Expected Return'] = pd.Series(self.expected_return)
#variance
        statistic_df['Variance'] = pd.Series(self.variance_dict)
#sharpe ratio
        for symbol in self.symbol_list[0:]:
            sharpe_ratio = (self.expected_return[symbol] - self.annual_risk_free_rate) / math.sqrt(self.variance_dict[symbol])
            self.sharpe_ratio_dict[symbol] = sharpe_ratio
        statistic_df['Sharpe Ratio'] = pd.DataFrame.from_dict(self.sharpe_ratio_dict, orient='index', columns=['Sharpe Ratio'])
#m2
        for symbol in self.symbol_list[0:]:
            m2 = (self.expected_return[symbol] - self.annual_risk_free_rate) * (math.sqrt(self.variance_dict[self.market_symbol]) / math.sqrt(self.variance_dict[symbol])) + self.annual_risk_free_rate
            self.m2_alpha_dict[symbol] = m2 - self.expected_market_return
        statistic_df['M2 Alpha'] = pd.DataFrame.from_dict(self.m2_alpha_dict, orient='index', columns=['M2 Alpha'])
# treynor ratio
# use actual returns for treynor ratio
        for symbol in self.symbol_list[0:]:
            treynor_ratio = (self.mean_dict[symbol] - self.annual_risk_free_rate) / self.beta_dict[symbol]
            statistic_df.loc[symbol, 'Treynor Ratio'] = treynor_ratio
#jensen's alpha -> based on the mean return
# jensens alpha has to use actual returns, it cannot use expected returns
# do not perform calculation for market symbol
        for symbol in self.symbol_list[0:]:
            if symbol == self.market_symbol:
                self.jensens_alpha_dict[symbol] = 0
            else:
                jensens_alpha = (self.mean_dict[symbol] - (self.annual_risk_free_rate + self.beta_dict[symbol] * (self.expected_market_return - self.annual_risk_free_rate)))
                self.jensens_alpha_dict[symbol] = jensens_alpha
        statistic_df['Jensen\'s Alpha'] = pd.DataFrame.from_dict(self.jensens_alpha_dict, orient='index', columns=['Jensen\'s Alpha'])
#non-systematic variance
        for symbol in self.symbol_list[0:]:
            nonsystematic_variance = self.variance_dict[symbol] - (self.beta_dict[symbol]**2 * self.variance_dict[self.market_symbol])
            self.nonsystematic_variance_dict[symbol] = nonsystematic_variance
            statistic_df.loc[symbol, 'Non-Systematic Variance'] = nonsystematic_variance
#information ratio
        for symbol in self.symbol_list[0:]:
            if symbol == self.market_symbol:
                self.information_ratio_dict[symbol] = 0
            else:
                information_ratio = self.jensens_alpha_dict[symbol] / self.nonsystematic_variance_dict[symbol]
                self.information_ratio_dict[symbol] = information_ratio
        statistic_df['Information Ratio'] = pd.DataFrame.from_dict(self.information_ratio_dict, orient='index', columns=['Information Ratio'])
#Check 1: is the sharpe ratio of the security greater than the market sharpe ratio * Correlation
        for symbol in self.symbol_list[0:]:
            if symbol == self.market_symbol:
                self.check1_dict[symbol] = True
            else:
                check1_beta = (correlation_matrix.loc[symbol, self.market_symbol] * (math.sqrt(self.variance_dict[symbol]) / math.sqrt(self.variance_dict[self.market_symbol])))
                check1_expected_return = self.annual_risk_free_rate + check1_beta * (statistic_df.loc[self.market_symbol, 'Expected Return'] - self.annual_risk_free_rate)
                check1_sharpe_ratio = (check1_expected_return - self.annual_risk_free_rate) / math.sqrt(self.variance_dict[symbol])
                if check1_sharpe_ratio > (statistic_df.loc[self.market_symbol, 'Sharpe Ratio'] * (correlation_matrix.loc[symbol, self.market_symbol])):
                    self.check1_dict[symbol] = True
                else:
                    self.check1_dict[symbol] = False
        statistic_df['Check 1'] = pd.DataFrame.from_dict(self.check1_dict, orient='index', columns=['Check 1'])
#Check 2: Is the Jensen Alpha Positive
#Optimal Portfolio Weighting
        sum_of_valid_information_ratios = 0
        sum_of_valid_alphas = 0
        sum_of_valid_nonsys_vars = 0
        #Create weightings for non-market portfolio candidates
        for symbol in self.symbol_list[0:]:
            if self.jensens_alpha_dict[symbol] > 0 and self.check1_dict[symbol] is True:
                sum_of_valid_information_ratios += self.information_ratio_dict[symbol]
        for symbol in self.symbol_list[0:]:
            if self.jensens_alpha_dict[symbol] > 0 and self.check1_dict[symbol] is True:
                    #this is the weight of the non-market security within only the portfolio of non-market securities
                    self.non_market_weightings_only_dict[symbol] = self.information_ratio_dict[symbol] / sum_of_valid_information_ratios
            else:
                self.non_market_weightings_only_dict[symbol] = 0
        for symbol in self.symbol_list:
            sum_of_valid_alphas += self.non_market_weightings_only_dict[symbol] * self.jensens_alpha_dict[symbol]
            sum_of_valid_nonsys_vars += self.non_market_weightings_only_dict[symbol]**2 * self.nonsystematic_variance_dict[symbol]
        proportional_weight_in_nonmarket_portfolio = sum_of_valid_alphas / sum_of_valid_nonsys_vars
        propotional_weight_in_market_portfolio = self.expected_return[self.market_symbol] / self.variance_dict[self.market_symbol]
        market_factor_weighting = propotional_weight_in_market_portfolio / proportional_weight_in_nonmarket_portfolio
        #we subtract 1 because we want to know how much in % form the market is greater than the non market
        market_factor_weighting = market_factor_weighting - 1
        #this calculation solves for the amount in the non-market portfolio
        allocation_non_market_portfolio = 1 / (1 + market_factor_weighting)
        allocation_market_portfolio = market_factor_weighting * allocation_non_market_portfolio
        #Calculate optimal portfolio weighting
        for symbol in self.symbol_list[0:]:
            if symbol == self.market_symbol:
                self.optimal_portfolio_weighting_dict[symbol] = allocation_market_portfolio
            else:
                if self.jensens_alpha_dict[symbol] > 0 and self.check1_dict[symbol] is True:
                    self.optimal_portfolio_weighting_dict[symbol] = (allocation_non_market_portfolio * self.non_market_weightings_only_dict[symbol])
                else:
                    self.optimal_portfolio_weighting_dict[symbol] = 0
        statistic_df['Optimized Weightings'] = pd.DataFrame.from_dict(self.optimal_portfolio_weighting_dict, orient='index', columns=['Optimized Weightings'])
# -------------------------------
# -------------------------------
#Start portfolio

        portfolio_row = pd.DataFrame(columns=statistic_df.columns)
#portfolio expected return
# 1 - Actual Weight of portfolio should be ivnested in the risk free rate!!!
# portfolio weight is the sum of all the securities in the portfolio
        portfolio_row.loc['Portfolio', 'Actual Weight'] = statistic_df['Optimized Weightings'].sum()
        for symbol in self.symbol_list[0:]:
            if symbol == self.symbol_list[0]:
                portfolio_expected_return = statistic_df.loc[symbol, 'Expected Return'] * statistic_df.loc[symbol, 'Optimized Weightings']
            else:
                portfolio_expected_return += statistic_df.loc[symbol, 'Expected Return'] * statistic_df.loc[symbol, 'Optimized Weightings']
        portfolio_expected_return += (1 - portfolio_row.loc['Portfolio', 'Actual Weight']) * self.annual_risk_free_rate
        portfolio_row.loc['Portfolio', 'Expected Return'] = portfolio_expected_return
#portfolio variance
#portfolio variance is calculated = W**2 * Var + W*W*COV
#We know Cash variance is 0. We also know Cash Correlation is 0.
#We know weight in cash is 1 - sum of weights in securities
#Therefore, we do not need to include cash in the variance calculation because its sum of covariance terms would = 0, and its sum of variance terms would also = 0.
        sum_of_covariance_terms = 0
        for i, symbol_1 in enumerate(self.symbol_list[0:]):
            for j, symbol_2 in enumerate(self.symbol_list[0:]):
                weight_1 = statistic_df.loc[symbol_1, 'Optimized Weightings']
                weight_2 = statistic_df.loc[symbol_2, 'Optimized Weightings']
                covariance = covariance_matrix.loc[symbol_1, symbol_2]
                if i != j:
                    sum_of_covariance_terms += weight_1 * weight_2 * covariance
        sum_of_variance_terms = 0
        for symbol in self.symbol_list[0:]:
            weight = statistic_df.loc[symbol, 'Optimized Weightings']
            variance = statistic_df.loc[symbol, 'Variance']
            sum_of_variance_terms += (weight**2) * variance
        portfolio_variance = sum_of_variance_terms + sum_of_covariance_terms
        portfolio_row.loc['Portfolio', 'Variance'] = portfolio_variance
#portfolio beta -> weighted average of holding betas
        portfolio_beta = 0
        for symbol in self.symbol_list[0:]:
            portfolio_beta += statistic_df.loc[symbol, 'Beta'] * statistic_df.loc[symbol, 'Optimized Weightings']
        portfolio_row.loc['Portfolio', 'Beta'] = portfolio_beta
#portfolio sharpe ratio
        portfolio_sharpe_ratio = (portfolio_expected_return - self.annual_risk_free_rate) / math.sqrt(portfolio_variance)
        portfolio_row.loc['Portfolio', 'Sharpe Ratio'] = portfolio_sharpe_ratio
#portfolio m2 alpha
        portfolio_m2 = (portfolio_expected_return - self.annual_risk_free_rate) * (math.sqrt(self.variance_dict[self.market_symbol]) / math.sqrt(portfolio_variance)) + self.annual_risk_free_rate
        portfolio_row.loc['Portfolio', 'M2 Alpha'] = portfolio_m2 - self.expected_market_return
#portfolio treynor ratio
        portfolio_treynor_ratio = (portfolio_expected_return - self.annual_risk_free_rate) / portfolio_beta
        portfolio_row.loc['Portfolio', 'Treynor Ratio'] = portfolio_treynor_ratio
#portfolio jensens alpha
        portfolio_jensens_alpha = (portfolio_expected_return - (self.annual_risk_free_rate + 1 * (self.expected_market_return - self.annual_risk_free_rate)))
        portfolio_row.loc['Portfolio', 'Jensen\'s Alpha'] = portfolio_jensens_alpha
        portfolio_row = portfolio_row.dropna(axis=1, how='any')
        statistic_df = pd.concat([statistic_df, portfolio_row])
        if return_portfolio_only:
            print('Portfolio Comparison to Market:')
            statistic_df = statistic_df.loc[[self.market_symbol, 'Portfolio']]
            return statistic_df.drop(columns=['Information Ratio'])
        
        return statistic_df

    def risk_return_tradeoff_test(self, portfolio_mode=False):
        # we also need to create a new optimized portfolio, so we need the weight of the securities included, which is information ratio
        # 1- weight is invested in risk free rate
        # I also want this to work for a given portfolio

        # Weight of a security in portfolio equals the Information Ratio of the security divided by the sum of all information ratios in the portfolio
        statistic_df = self.calculate_statistics()
        correlation_matrix = self.generate_correlation_matrix(print_on=False, return_on=True)
        covariance_matrix = self.generate_covariance_matrix(print_on=False, return_on=True)
        risk_free_rate = .02
        for symbol in self.symbol_list[0:]:
            if symbol == self.market_symbol:
                continue
            else:
                check1_beta = (correlation_matrix.loc[symbol, self.market_symbol] * (math.sqrt(self.variance_dict[symbol]) / math.sqrt(self.variance_dict[self.market_symbol])))
                check1_expected_return = risk_free_rate + check1_beta * (statistic_df.loc[self.market_symbol, 'Expected Return'] - risk_free_rate)
                check1_sharpe_ratio = (check1_expected_return - risk_free_rate) / math.sqrt(self.variance_dict[symbol])
                if check1_sharpe_ratio > (statistic_df.loc[self.market_symbol, 'Sharpe Ratio'] * (correlation_matrix.loc[symbol, self.market_symbol])):
                    print(f"{symbol}:") 
                    print(f"Passed Check 1: New Sharpe Ratio is Greater Than Previous Sharpe Ratio * Correlation")
                    print(f"New Expected Return: {check1_expected_return:.4f}, Old Expected Return: {statistic_df.loc[self.market_symbol, 'Expected Return']:.4f}")
                    print(f"New Sharpe Ratio: {check1_sharpe_ratio:.4f}, Old Sharpe Ratio: {statistic_df.loc[self.market_symbol, 'Sharpe Ratio']:.4f}")
                    if self.jensens_alpha_dict[symbol] > 0:
                        
                        print(f'Passed Check 2: Jensen\'s Alpha is positive')
                        print("--------------------------------------------------")
                    else:
                        print(f"Failed Check 2: Jensen\'s Alpha is Negative")
                        print(f'It is therefore a Detrimental Addition to the Portfolio.')
                        print("--------------------------------------------------")
                else:
                    print(f"{symbol}: Detrimental Addition to Portfolio")
                    print("--------------------------------------------------")
        
        
            
        
    
    def return_df(self):
        #show all rows
        pd.set_option('display.max_rows', None)
        return self.df