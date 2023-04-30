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
    best_score=0
    architecture=[32, 16], [32, 32], [16, 8], [64, 32, 16]
    best_network=None
    for epoch in range(5, 200):
        for params in architecture:
            model = build_model(params, train_x)
            history= model.fit(train_x, train_y, epochs=epoch,
                      batch_size=32,validation_split=0.0, verbose=0)
            # pd.DataFrame(history.history).plot()
            # plt.show()
            trial_ANN_predicted=model.predict(test_x,verbose=0)
            score=sklearn.metrics.r2_score(test_y, trial_ANN_predicted)
            if score>best_score:
                best_epoch=epoch
                best_model=model
                ANN_predicted=trial_ANN_predicted
                best_score=score
                best_network=params
        if verbose:
            print('epoch:',epoch,'best:',best_score, 'best_network:',best_network)

    plt.plot(ANN_predicted, test_y, "bo")
    plt.plot(range(2),range(2),'r')
    plt.xlim((0,0.3))
    plt.ylim((0,0.3))
    
    return best_network, best_epoch, best_score, best_model


best_network, best_epoch, best_score, best_model=optimal_ANN(verbose=1)

# saved the model with the best R2
best_model.save('BestANN_raw_data_80_20_split_yu')


def optimal_svm(verbose=True):

    c_range=[0.1, 1, 10, 100, 500]
    gamma_range=[1, 0.1, 0.01, 0.001, 0.0001]
    eps_range=[0.1, 0.1, 0.01, 0.001, 0.0001]
    best_score=0
    best_model, best_params=None, None
    for c in c_range:
        for g in gamma_range:
            for e in eps_range:
                svm_reg = svm.SVR(C=c, kernel='rbf',epsilon=e,gamma=g)
                svm_reg.fit(train_x, np.asarray(train_y).flatten())
                svm_predict=svm_reg.predict(test_x)
                score=sklearn.metrics.r2_score(test_y, list(svm_predict))
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
