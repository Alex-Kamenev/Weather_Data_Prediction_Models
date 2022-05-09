%load_ext autoreload
%autoreload 2

# imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Modelpredict:
    #define class variables
    
    #feature engineering methods - many one or more - will became clear from research on the tropic
    #out data is pretty good so there may not be much to do here. Things like converting to numeric and one hot encoding 
    
    #method to normalize dataset
    def normalize(self, df):
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df)
        return data_scaled
        
    
    #method to standardization
    def standardize(self, df):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
        return data_scaled
    
    #feature selection methods - will probably have to search for a good lib for this task and use it 
    #one such lib is Boruta
    def one_hot_encode(self, df, feature):
        dummy_gender = pd.get_dummies(df[feature], prefix=feature+'_')
        df = pd.merge(
        left=df,
        right=dummy_gender,
        left_index=True,
        right_index=True,
        )
        df.drop([feature], axis=1, inplace=True)
        return df
    
    #methods to split the dataset for x and y and train and test
        #allow for inputs on the spliut details in the methd call itself
    def split(self, df, target_feature, test_size):
        X = df.drop([target_feature], axis=1)
        y = df[target_feature]
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state = 21)
        return X_train, X_test, Y_train, Y_test;
        
        
    #hethods to handle resampling - so that we can deal with imbalanced datasets (train and test)
    
    #Method to define models
    
        # allow for fit and fit&transform
        
        #do not couple fit and predict on one method should be able to call a method that does fitting and 
        #nother method that does the predict and potentially displays some metrics related to the outcome of the predicition
    
        #basic structure of a method for a model
        #the method should be able to accept the X and y sets as paramiters
        #there may be other paramiters the method accepts - please make choises based on the model 
        #you are working on
        #the out put of the method should be such that we can use it in later method calls if need be
        
    #Linear regression
        #Possibly try different regression models
    def linear_reg(self, df, feature, test_size):

        # Split into a training and testing set
        X_train, X_test, Y_train, Y_test = self.split(df, feature, test_size)

        # Define the pipeline for scaling and model fitting
        pipeline = Pipeline([
            ("StandardScaler", StandardScaler()),
            ("MinMaxScaler", MinMaxScaler()),
            ("Linear Regression", LinearRegression())
        ])

        # Scale the data and fit the model
        pipeline.fit(X_train, Y_train)

        # Evaluate the model
        Y_pred = pipeline.predict(X_test)
        
        # display metrics relavent to the model's performance
        print('Pipeline Linear Regression Coef')
        cdf = pd.DataFrame(pipeline.named_steps['Linear Regression'].coef_, X_train.columns, columns=['Coeff'])
        display(cdf)
        print('Pipeline Mean Squered Error: ', mean_squared_error(Y_pred, Y_test))
        print('Pipeline Mean Absolute Error: ', mean_absolute_error(Y_pred, Y_test))
        print('Pipeline Score', pipeline.score(X_test, Y_test))
        
        # get a df with the actual vs predicted values        
        result = self.make_results_df(Y_pred, Y_test)
        display(result.head(10))
        
        # plot a scatter plot with the actual vs predicted values          
        self.plot_scatter(Y_pred, Y_test, feature)
        
        # plot kde plot for actual and predicted values
        self.plot_kde( Y_pred, Y_test)
        
        return Y_pred, Y_test
        
        
    #Random forest model
    def random_forest(self, df, feature, test_size):

        # Split into a training and testing set
        X_train, X_test, Y_train, Y_test = self.split(df, feature, test_size)

        # Define the pipeline for scaling and model fitting
        pipeline = Pipeline([
            ("StandardScaler", StandardScaler()),
            ("MinMaxScaler", MinMaxScaler()),
            ('clf', RandomForestRegressor())
        ])
        
        # Declare a hyperparameter grid
        param_grid = {
            "clf__n_estimators": [100, 400, 1200],
            "clf__max_depth": [10, 40, 120, None],
            "clf__max_features": ['auto', 'sqrt'],
            "clf__bootstrap": [True, False],
            "clf__min_samples_leaf": [1, 2],
            "clf__min_samples_split": [2, 5],
        }

        # Perform grid search, fit it, and print score
        gs = GridSearchCV(pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1000)
        
        gs.fit(X_train, Y_train)

        # Evaluate the model
        best_random = gs.best_estimator_
        
        Y_pred = best_random.predict(X_test)
        
        # display metrics relavent to the model's performance
        print('Pipeline Best params:\n', gs.best_params_)
        
        print('Pipeline Mean Squered Error: ', mean_squared_error(Y_pred, Y_test))
        print('Pipeline Mean Absolute Error: ', mean_absolute_error(Y_pred, Y_test))
        print('Pipeline Score', gs.score(X_test, Y_test))
        
        # get a df with the actual vs predicted values        
        result = self.make_results_df(Y_pred, Y_test)
        display(result.head(10))
        
        # plot a scatter plot with the actual vs predicted values          
        self.plot_scatter(Y_pred, Y_test, feature)
        
        # plot kde plot for actual and predicted values
        self.plot_kde( Y_pred, Y_test)
        
        return Y_pred, Y_test
        
    
    #methods to take the outcome of the modeling related methods and produce some visual representation of the performance of the model
    
    def make_results_df(self, Y_pred, Y_test):
        result = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
        return result

    def plot_scatter(self, Y_pred, Y_test, feature):
        plt.figure(figsize=(16, 12))
        plt.xlabel(feature, fontsize=18)
        plt.ylabel(feature, fontsize=18)
        plt.scatter(Y_test, Y_pred, label="PREDICTED VALUE")
        plt.scatter(Y_test, Y_test, label="ACTUAL VALUE")
        plt.legend(prop={'size': 14})
        plt.show()
        
    def plot_kde(self, Y_pred, Y_test):    
        plt.figure(figsize=(14, 11))
        sns.set(font_scale = 2)
        sns.kdeplot(Y_pred)
        sns.kdeplot(Y_test)
        plt.legend(labels=["Predicted Value", "Actual Value"], prop={'size': 16})
        
    def ts_plot_outcome(data_train, data_test, predictions, feature, steps):
        # Plot

        fig, ax = plt.subplots(figsize=(20, 10))
        data_train[feature].plot(ax=ax, label='train')
        data_test[feature].plot(ax=ax, label='test', color="darkorange")
        predictions.plot(ax=ax, label='predictions', color="forestgreen")
        plt.title("The Whole Set")
        ax.legend();

        fig, ax = plt.subplots(figsize=(20, 10))
        data_train.iloc[-steps:][feature].plot(ax=ax, label='train')
        data_test[feature].plot(ax=ax, label='test', color="darkorange")
        predictions.plot(ax=ax, label='predictions', color="forestgreen")
        plt.title("The Last {} Records of Train and All of Test".format(steps))
        ax.legend();

        fig, ax = plt.subplots(figsize=(20, 10))
        data_test[feature].plot(ax=ax, label='test', color="darkorange")
        predictions.plot(ax=ax, label='predictions', color="forestgreen")
        plt.title("Test Only")
        ax.legend();
        
        # Test error
        error_mse = mean_squared_error(y_true = data_test[feature], y_pred = predictions)
        display(f"Test error (mse): {error_mse}")
    #Potential comparison method to show how all our models performed
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    