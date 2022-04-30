# imports


class Modelpredict:
    #define class variables
    
    #feature engineering methods - many one or more - will became clear from research on the tropic
    #out data is pretty good so there may not be much to do here. Things like converting to numeric and one hot encoding 
    
    #method to normalize dataset
    
    #method to standardization
    
    #feature selection methods - will probably have to search for a good lib for this task and use it 
    #one such lib is Boruta
    
    #methods to split the dataset for x and y and train and test
        #allow for inputs on the spliut details in the methd call itself
        
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
        
        #Random forest model
            #again do research and see about different models of this family
        
        #k-nearest neighbors 
        
        #Time series prediction models
        #Do some proper research
        
    
    #methods to take the outcome of the modeling related methods and produce some visual representation of the performance of the model
    
    #Potential comparison method to show how all our models performed
    