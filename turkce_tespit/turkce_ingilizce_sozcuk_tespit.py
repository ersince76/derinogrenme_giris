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

def harf_vektor(cumle):
    harfler = "abcçdefgğhıijklmnoöprsştuüvyzqwx"
    result =[]
    for h in harfler:
        value = round( cumle.count(h) / len( cumle) , 3)
        result.append( value )
    return np.array(result)

import time 

X = data[ 160000:160020 ,0:32]
y = data[ 160000:160020 ,32]

_, accuracy = model.evaluate(X, y)
print('Başarım: %.2f' % (accuracy*100)) 

"""
a = time.time()
# satır satır modeli işletmek 
for i,d in enumerate(X):
    yp = model.predict([[ d ]] )
    print( "beklenen= {}, tahmin edilen ={} , yorum ={} ".format(y[i], yp[0][0], round(yp[0][0]) ))

b = time.time()

print( "hesaplama suresi {}".format(str( (b -a) / len(y) ) ))

#%%
sozluk  = open("./turkce.txt")
#sozluk  = open("./english.txt")


tur = 0
eng = 0
i =0
for w in sozluk:
    i += 1
    if i % 1000 == 0:
        print( tur / (tur + eng))
        
    p = model.predict([[harf_vektor(w)]])
    
    if p < 0.5:
        eng += 1
        #print( w )
    else:
        tur += 1
        
    if eng == 50000:
        break
    #break

print ( tur / (tur + eng))
"""
