import numpy as np
import csv
import os
import sys
import pandas as pd
sys.path.append(".")

np.set_printoptions(legacy='1.25')
#import a daily dataframe with raw prices - no return calculation performed
#convert to yearly and perform return calculation
#if multiple tickers, keep all tickers

class Converter:
    def __init__(self, daily_df, symbol_list):
        self.daily_df = daily_df
        self.symbol_list = symbol_list
        self.yearly_df = self.convert_to_yearly()

    def convert_to_yearly(self):
        if self.symbol_list == []:
            raise ValueError("Symbol list is empty.")
        for symbol in self.symbol_list:
            self.daily_df[f'{symbol} Return'] = (self.daily_df[f'Close {symbol}'] - self.daily_df[f'Close {symbol}'].shift(252)) / self.daily_df[f'Close {symbol}'].shift(252)
        yearly_df = self.daily_df
        return yearly_df