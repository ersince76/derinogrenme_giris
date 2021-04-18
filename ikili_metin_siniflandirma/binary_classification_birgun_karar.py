"""
Temel Model 
"""

import preprocess_util
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold



# veri seti yukle
triple_dictionary, text, label = preprocess_util.load_trigram_dataset()


# seed 1923 ile sabitlenir
seed = 1923
np.random.seed(seed)

# girdi X ve Y olarak tanımlanıyor
X = text

encoder = LabelEncoder()
encoder.fit(label)
Y = encoder.transform(label)

#%%
# temel model
def create_baseline():
    # model olustur
    model = Sequential()
    model.add(Dense(100, input_dim=len(triple_dictionary)+1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # model derle
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#%%
# veri seti ile kfold
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=500, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("temel model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



"""
model = create_baseline()
model.fit(X, np.array(Y), validation_split=0.20,  epochs=200, verbose=2, batch_size = 100)

#%%
result=model.predict(preprocess_util.prepare_triple_input("karagümrük stadını istiyor", triple_dictionary))

print( result )
"""