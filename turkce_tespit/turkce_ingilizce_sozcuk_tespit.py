#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:07:07 2020

@author: ersince
"""

#%%

from keras.models import Sequential
from keras.layers import Dense  
import numpy as np 
data = np.genfromtxt("./sozluk_harf.csv", dtype=float, delimiter=',', skip_header=True)


#%%

np.random.shuffle(data)

# girdi(x) ve çıktı(y) belirle
X = data[: 160000 ,0:32]
y = data[: 160000,32]

#%%
# keras modeli hazırla
model = Sequential()
model.add(Dense(36, input_dim=32, activation='relu'))  
model.add(Dense(12, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))

# modeli derle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# modeli eğit
model.fit(X, y, epochs=100, verbose=1, batch_size = 32)

# keras modeli işlet
_, accuracy = model.evaluate(X, y)
print('Başarım: %.2f' % (accuracy*100)) 

#%%

X = data[ 160000:160020 ,0:32]
y = data[ 160000:160020 ,32]

_, accuracy = model.evaluate(X, y)
print('Başarım: %.2f' % (accuracy*100))
