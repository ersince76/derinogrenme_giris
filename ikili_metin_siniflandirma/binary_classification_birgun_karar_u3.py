"""
Temel Model

Guncelleme 1, StandardScaler ve Pipeline

Guncelleme 3, Daha büyük ağ [ 100 update 100->50->1 ]

"""

import preprocess_util
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

#upgrade1
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# seed 1923 ile sabitlenir
seed = 1923
np.random.seed(seed)

# veri seti yukle
triple_dictionary, text, label = preprocess_util.load_trigram_dataset()

# girdi X ve Y olarak tanımlanıyor
X = text #(4388, 1, 7254)
X = X.reshape(4388,7254)

encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)

# temel model
def create_baseline():
    # model olustur
    model = Sequential()
    model.add(Dense(100, input_dim=len(triple_dictionary)+1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # model derle
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#%%
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Daha Büyük Ağ: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


