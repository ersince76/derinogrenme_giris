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

from sklearn.model_selection import train_test_split

import numpy as np 


seed = 100
np.random.seed( seed )


#%%

data = np.genfromtxt("./turkce_ingilizce_harf_yogunluk.csv", dtype=float, delimiter=',', skip_header=True)

np.random.shuffle( data )

# girdi(x) ve çıktı(y) belirle
X = data[: 160000 ,0:32]
y = data[: 160000,32]


# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

#%%
# keras modeli hazırla
model = Sequential()
model.add(Dense(5, input_dim=32, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# modeli derle
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# modeli eğit
model.fit(X, y, validation_data=(X_test,y_test),  epochs=200, verbose=1, batch_size = 100)


#%%
"""
X = data[ 160000: ,0:32]
y = data[ 160000: ,32]
"""

# modeli dene
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


print( "Turkce" if model.predict([[ harf_vektor("Bütün insanlar hür, haysiyet ve haklar bakımından eşit doğarlar. Akıl ve vicdana sahiptirler ve birbirlerine karşı kardeşlik zihniyeti ile hareket etmelidirler.")]])[0,0] > 0.5 else "İngilizce"  )

print( "Turkce" if model.predict([[ harf_vektor("All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.")]])[0,0] > 0.5 else "İngilizce"  )

