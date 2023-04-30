import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def build_model(hiddenLayerSizes, train_data):
    """
    This function builds an ANN sequential model given the hidden layer sizes and training data.
    
    Returns the compiled model.
    """

    model =tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hiddenLayerSizes[0],input_dim=train_data.shape[1],activation='relu'))
    
    
    if len(hiddenLayerSizes)>1:
        for i in range(len(hiddenLayerSizes)-1):
            model.add(tf.keras.layers.Dense(hiddenLayerSizes[i+1],activation='relu')) #hidden layer generator loop
    else:
        pass
    
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mse', metrics=['mae'],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))  

    return model



def optimal_ANN(verbose=True):
    """
    This function computes the optimal ANN architecture. Architectures are defined inside.
    
    Returns the best model with relevant informations.
    
    """
    #initialize
    best_score=0
    best_network=None
    
    #the hidden layer neurons considered
    architecture=[32, 32], [32, 16, 8], [32, 32, 32], [16, 8, 4], [64, 32, 16]
    
    for epoch in range(5, 500):
        for params in architecture:
            #create a model
            model = build_model(params, train_x)
            #train the model
            history= model.fit(train_x, train_y, epochs=epoch,
                      batch_size=32,validation_split=0.0, verbose=0)
            #predict the testing data
            trial_ANN_predicted=model.predict(test_x,verbose=0)
            #compute the R2 value
            score=sklearn.metrics.r2_score(test_y, trial_ANN_predicted)
            #save the best model
            if score>best_score:
                best_epoch=epoch
                best_model=model
                ANN_predicted=trial_ANN_predicted
                best_score=score
                best_network=params
        if verbose:
            print('epoch:',epoch,'best:',best_score, 'best_network:',best_network)
    
    #plot the best model crossplot
    plt.plot(ANN_predicted, test_y, "bo")
    plt.plot(range(2),range(2),'r')
    plt.xlim((0,0.4))
    plt.ylim((0,0.4))
    
    return best_network, best_epoch, best_score, best_model

#run the optimization function 
best_network, best_epoch, best_score, best_model=optimal_ANN(verbose=1)

# saved the model with the best R2
best_model.save('BestANN_raw_data_80_20_split_yu')


def optimal_svm(verbose=True):
    """
    This function computes the optimal SVM hyperparameters.
    
    Returns the best model.
    
    """
    #define the ranges for the hyperparameters
    c_range=[0.1, 1, 10, 100, 500]
    gamma_range=[1, 0.1, 0.01, 0.001, 0.0001]
    eps_range=[0.1, 0.1, 0.01, 0.001, 0.0001]
    
    #initialize
    best_score=0
    best_model, best_params=None, None
    for c in c_range:
        for g in gamma_range:
            for e in eps_range:
                #compile
                svm_reg = svm.SVR(C=c, kernel='rbf',epsilon=e,gamma=g)
                #train the model
                svm_reg.fit(train_x, np.asarray(train_y).flatten())
                #predict for testing data
                svm_predict=svm_reg.predict(test_x)
                #compute the R2 value
                score=sklearn.metrics.r2_score(test_y, list(svm_predict))
                #save the best model
                if score>best_score:
                    best_model=svm_reg
                    best_score=score
                    best_params=[c, g, e] #C, gamma, epsilon
                    best_prediction=svm_predict
        if verbose:
            print('best_score:',best_score,'best_params:',"C:",best_params[0],'G:',best_params[1],'e:',best_params[2])
    plt.plot(best_prediction, test_y, "bo")
    plt.plot(range(2),range(2),'r')
    plt.xlim((0,0.3))
    plt.ylim((0,0.3))
    
    return best_model

svm_optimal_model=optimal_svm()
