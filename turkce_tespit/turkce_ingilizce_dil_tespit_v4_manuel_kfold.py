#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:07:07 2020

@author: ersince
"""

#%%
# kütüphaneleri bildir, rastgele sayı üreteci çekirdeğini 100 belirle
from keras.models import Sequential
from keras.layers import Dense  

from sklearn.model_selection import StratifiedKFold

import numpy as np 


seed = 100
np.random.seed( seed )


#%%

data = np.genfromtxt("./turkce_ingilizce_harf_yogunluk.csv", dtype=float, delimiter=',', skip_header=True)

np.random.shuffle( data )

# girdi(x) ve çıktı(y) belirle
X = data[: 160000 ,0:32]
y = data[: 160000,32]


# 10-fold cross validation test oluştur
kfold = StratifiedKFold(n_splits=10, shuffle=True,   random_state=seed)
cvscores = []

#%%
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=32, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Fit the model
    model.fit(X[train], y[train], epochs=200, batch_size=100, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))