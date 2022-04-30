import pandas as pd
import missingno as msno
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class DataInfo:
    _df = None
    
    def __init__(self, df):
        self._df = df
    
    def describe(self):
        display(self.head())
        display(self._df.describe())
        display(self.info())
        return
    
    def head(self):
        return self._df.head()
    
    def info(self):
        return self._df.info()
    
    def nulls(self):
        print(f'\nShow number of missing instances per feature \n' + '--'*10 )
        display(self.isnull())
        print(f'\nShow number of present instances per feature \n' + '--'*10 )
        display(self.notnull())
        display(self.check_missing_values_custom())
        return
    
    def isnull(self):
        return self._df.isnull().sum()
    
    def notnull(self):
        return self._df.notnull().sum()
    
    def show_missing(self, *argv):
        for arg in argv:
            print("\nFeature name:", arg)
            display(self.show_missing_per_column(arg))
    
    def show_missing_per_column(self, column_name):
        return self._df[self._df[column_name].isnull()]
    
    def get_column_names_where_missing_data(self):
        return self._df.columns[self._df.isnull().any()]
    
    def check_missing_values_custom(self):
        # get number of features and records
        print("Shape", self._df.shape)
        # get names of columns and place in list
        list_name = self._df.columns.to_list()
        for i in list_name:
            item_counts = self._df[i].value_counts()
            items = self._df[i].unique()
            print("\n" + '** '*10 )
            print("\nFeature:", i)
            print("\nNumber of unique values:", len(items),"\n")
            print(items)
            print("_ _ _ _ _ _ _ _ _ _ _ _ _ _\n")
            print(item_counts)
            print("___________________________\n")
        return
    
    def missing_bar_plot(self):
        display(msno.bar(self._df, figsize=(12, 8)))
        display(msno.matrix(self._df, figsize=(12, 8)))
        display(msno.heatmap(self._df, figsize=(12, 8)))
        return
    
    def to_date_time(self, name):
        self._df[name] = pd.to_datetime(self._df[name])
        return
    
    def as_string(self, *argv):
        for arg in argv:
            self._df[arg] = self._df[arg].astype("string")
        return
    
    def average_fill_in_missing(self, list_of_features):
        for feature in list_of_features:
            for date in self._df.loc[self._df[feature].isnull(), 'DATE']:
                present = self._df.loc[(self._df[feature].notnull()) & (self._df['DATE'] == date), feature]
                new_value = present.sum() / present.count()
                self._df.loc[(self._df[feature].isnull()) & (self._df['DATE'] == date), feature] = new_value
        return
    
    def half_sum_fill_in_missing(self, *argv):
        for arg in argv:
            for date in self._df.loc[self._df[arg].isnull(), 'DATE']:
                self._df.loc[self._df['DATE'] == date, "TAVG"] = ((self._df.loc[self._df['DATE'] == date, "TMIN"] +
                                                  self._df.loc[self._df['DATE'] == date, "TMAX"]) / 2)
        return
    
    def group_by(self, feature):
        groups = self._df.groupby([feature])
        list_of_groups = list(groups)
        return list_of_groups
    
    def get_unique(self, column):
        return self._df[column].unique()
    
    def get_unique_df(self, df, column):
        return df[column].unique()
    
    def get_list_dfs(self, list_of_values, column):
        df_list = []
        for i, feature in enumerate(list_of_values):
            df_list.append(self._df[self._df[column] == feature])
        return df_list
    
    def get_list_dfs_df(self, df, list_of_values, column):
        df_list = []
        for i, feature in enumerate(list_of_values):
            df_list.append(df[df[column] == feature])
        return df_list

    def feature_extraction_date(self):
        self._df['YEAR'], self._df['MONTH'], self._df['DAY'] = self._df['DATE'].dt.year, self._df['DATE'].dt.month, self._df['DATE'].dt.day
        return
    
    def feature_extraction_days_since_jan_1(self):
        self._df['DAYS_SINCE_JAN_1'] = 0
        for i, row in self._df.iterrows():
            value = row['DATE'].timetuple().tm_yday - 1
            self._df.loc[i, 'DAYS_SINCE_JAN_1'] = value
    
    def get_list_of_lists_df_split(self, list_of_dfs, feature):
    
        df_i_list_of_dfs_j_features_df = []

        for df in list_of_dfs:    
            # extract current state of dataframe

            # get unique feature list from dataset
            list_of_feature = self.get_unique_df(df, feature)

            # get list of dataframes split on feature
            df_i_list_of_dfs_j_features_df.append(self.get_list_dfs_df(df, list_of_feature, feature))

        return df_i_list_of_dfs_j_features_df

    
        
    def get_df(self):
        return self._df
    
    def set_df(self, df):
        self._df = df
        return

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
