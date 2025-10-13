import numpy as np

np.set_printoptions(legacy='1.25')

#Goals
# Usable in any model
# Take in a dataframe and return a randomized sampled data frame
class Sampler:
    def __init__(self, df, sample_size=30):
        self.df = df
        self.sample_size = sample_size
        self.sampled_df = self.generate_sampled_df()

    def generate_sampled_df(self):
        if self.sample_size >= len(self.df):
            return self.df.copy()
        else:
            sampled_indices = np.random.choice(self.df.index, size=self.sample_size, replace=False)
            sampled_df = self.df.loc[sampled_indices].sort_index()
            return sampled_df