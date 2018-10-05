import os
#Ako nisu dodani CUDNN DLL-ovi u PATH i postoji vise grafickih kartica 
os.environ['PATH'] = os.environ['PATH'] + ";C:\\Users\\tmanojlo\\cuda\\bin"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import KFold


X,Y = load_dataset("C:\\users\\tmanojlo\\Desktop\\music_dataset\\")

#500 prvih znacajki da se ubrza treniranje
X = X[:,0:500]

X = preprocessing.scale(X)

discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
X = discretizer.fit_transform(X)

X = to_categorical(X)

kf = KFold(n_splits=10)
kf.get_n_splits(X)

scores_list = []

for train_index, test_index in kf.split(files):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(LSTM(100,input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5, batch_size = 10)

    scores = model.evaluate(X_test, y_test, verbose=0)
    scores_list.append(scores[1]*100)
    

score = np.mean(scores_list)
print("Accuracy: %.2f%%" % (score))



