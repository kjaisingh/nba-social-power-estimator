#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:17:47 2019

@author: jaisi8631
"""

# imports
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# create data structures
dataRaw = pd.read_csv('dataset.csv')
dataTwitter = pd.read_csv('data/nba_2017_twitter_players.csv')
rawX = dataRaw.iloc[:, :].values
rawY = dataTwitter.iloc[:, :].values
data = []


# add salary data to raw player data
for i in range(0, rawX.shape[0]):
    index = -1
    j = 0
    while(index == -1 and j < rawY.shape[0]):
        if(rawX[i][0] == rawY[j][0]):
            index = j
        j += 1
    if(index != -1):
        x = rawX[j, 1:].tolist()
        retweet = rawY[index][2]
        if(retweet == 'nan' or retweet is None):
            retweet = 0
        x.append(retweet)
        data.append(x)


# create training matrix with values
data = np.array(data)
data[np.isnan(data)] = 0
data = data.astype(float)
X = data[:, :-1]
y = data[:, -1]
yTrue = y
y.shape = (y.size, 1)


# one hot encode and update training matrix
encode = X[:, 0]
encode = encode.astype(int)
n_values = np.max(encode) + 1
encode = np.eye(n_values)[encode]
encode = encode[:, 1:]
X = np.delete(X, 0, 1)
X = np.append(encode, X, axis=1)


# scale data 
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


# split data using a 4:1 ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# create sequential regression model
model = Sequential()
model.add(Dense(1000, input_dim = len(X_train[0]), 
                kernel_initializer = 'random_uniform', activation = 'sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation = 'linear'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal'))


# compile and execute
adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'mean_absolute_error', optimizer = adam)
model.fit(X_train, y_train, batch_size = 1, 
          nb_epoch = 500)


# make predictions
y_pred = model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(yTrue, y_pred)


# output error score
print("The mean absolute error of the model is: " + str(score))


# save model
model.save('my_model.h5')