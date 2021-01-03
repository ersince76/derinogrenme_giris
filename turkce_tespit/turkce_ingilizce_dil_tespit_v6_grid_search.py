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
from sklearn.model_selection import GridSearchCV

import numpy as np 


seed = 100
np.random.seed( seed )

#%%
def create_model(optimizer= 'rmsprop' , init= 'glorot_uniform'):
    model = Sequential()
    model.add(Dense(5, input_dim=32, kernel_initializer=init, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
    


#%%

data = np.genfromtxt("./turkce_ingilizce_harf_yogunluk.csv", dtype=float, delimiter=',', skip_header=True)

np.random.shuffle( data )

# girdi(x) ve çıktı(y) belirle
# cok cok uzun bir zaman gerektiği için :) 0.01 e indirdik
X = data[: 1600 ,0:32]
y = data[: 1600,32]



#
model= KerasClassifier(build_fn=create_model, verbose=1)

#grid search epochs, batch size ve optimizer
optimizers = [ "rmsprop" , "adam" ]
inits = [ "glorot_uniform" , "normal" , "uniform" ]
epochs = [50, 100, 150]
batches = [5, 10, 20]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)

grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_[ "mean_test_score" ]
stds = grid_result.cv_results_[ "std_test_score" ]
params = grid_result.cv_results_[ "params"]
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))