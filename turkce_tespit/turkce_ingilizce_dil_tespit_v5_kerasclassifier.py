#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ersince
"""

#%%
# kütüphaneleri bildir, rastgele sayı üreteci çekirdeğini 100 belirle
from keras.models import Sequential
from keras.layers import Dense  

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import numpy as np 


seed = 100
np.random.seed( seed )

#%%
def create_model():
    model = Sequential()
    model.add(Dense(5, input_dim=32, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return model
    


#%%

data = np.genfromtxt("./turkce_ingilizce_harf_yogunluk.csv", dtype=float, delimiter=',', skip_header=True)

np.random.shuffle( data )

# girdi(x) ve çıktı(y) belirle
X = data[: 160000 ,0:32]
y = data[: 160000,32]



#
model= KerasClassifier(build_fn=create_model, epochs=200, batch_size=100, verbose=0)

# 10-fold cross validation test oluştur
kfold = StratifiedKFold(n_splits=10, shuffle=True,   random_state=seed)


results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

