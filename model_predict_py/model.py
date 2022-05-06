# imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics

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
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
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
        
        print('Pipeline Linear Regression Coef')
        cdf = pd.DataFrame(pipeline.named_steps['Linear Regression'].coef_, X_train.columns, columns=['Coeff'])
        display(cdf)
        print('Pipeline Mean Squered Error: ', mean_squared_error(Y_pred, Y_test))
        print('Pipeline Mean Absolute Error: ', mean_absolute_error(Y_pred, Y_test))
        print('Pipeline Score', pipeline.score(X_test, Y_test))
        
        
        return Y_pred, Y_test
        
        
        #Random forest model
            #again do research and see about different models of this family
        
        #k-nearest neighbors 
        
        #Time series prediction models
        #Do some proper research
        
    
    #methods to take the outcome of the modeling related methods and produce some visual representation of the performance of the model
    
    def make_results_df(self, Y_pred, Y_test):
        result = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
        return result
    
    def plot_y_test_train(self, Y_pred, Y_test):
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(0, len(Y_pred), len(Y_pred)), Y_pred, label = "Forecast Value")
        plt.plot(np.linspace(0, len(Y_pred), len(Y_pred)), Y_test, label = "Actual Value")
        plt.legend()
        plt.show()
        
    #Potential comparison method to show how all our models performed
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    