#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
"""
Para el siguiente entrenamiento se usara las librerias de keras
, se intento el uso con SVC pero el modelo tardo demasiado en corrrer usando
los 16 GB de ram de la computadora.
"""
import tensorflow as tf
from keras.utils import np_utils  # representa las etiquetas en one-hot para kera
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import SGD
"""
Se cargara las
"""

data = pd.read_csv('./train_labels.csv')
# se ordenara los datos segun el label corespondiente
data_sort = data.sort_values(['label'])
y = data_sort.label.values
id = data_sort.id.values

# se intenta convir1tiendo las imagenes a grises
X_list = []
for i in id:
    a = cv2.imread('./train/{}.tif'.format(id[0]))  # se lee las imagenes
    b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)  # se cambia a escala de grises
    c = b[31:31+32, 31:31+32]  # se corta la imagen a 32 x 32
    # d = c.flatten()  # vconvierte un array 2d a 1d
    X_list.append(c)

X = np.array(X_list)
print("imagens a usar: "+str(X.shape)+"\n"+"labels = "+str(y.shape))
X = np.array(X/255.0, dtype='float32')  # se normaliza los datos
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
X.shape
nc = 2
Y = np_utils.to_categorical(y, num_classes=nc)

x_train, x_valid, y_train, y_valid = train_test_split(X, Y,
                                                      train_size=0.9,
                                                      random_state=42)


model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5),
                 activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(5, 5),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(nc, activation='softmax'))

sgd = SGD(lr=0.1)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

epocas = 10
tamanho_lote = 128
model.fit(x_train, y_train, epochs=epocas, batch_size=tamanho_lote,
          validation_data=(x_valid, y_valid))

model_jason = model.to_json()
with open("modelo.json", 'w') as j_f:
    j_f.write(model_jason)

model.save_weights("modelo.hi5")
print("guardado en el disco")
