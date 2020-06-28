#!/usr/bin/env python3

from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import pandas as pd
import cv2
"""
Para el siguiente entrenamiento se usara las librerias de keras
, se intento el uso con SVC pero el modelo tardo demasiado en corrrer usando
los 16 GB de ram de la computadora.
"""
import tensorflow as tf
# representa las etiquetas en one-hot para kera
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, SeparableConv2D
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

"""
Se crea un modelo tentativo
"""


def ajuste_modelo(width, height, depth, classes):

    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    # CONV => RELU => POOL
    model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU => POOL) * 2
    model.add(SeparableConv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(SeparableConv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU => POOL) * 3
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.summary()
    # return the constructed network architecture
    return model


"""
SE leera las imagenes
"""
data = pd.read_csv('./train_labels.csv')
# se ordenara los datos segun el label corespondiente
data_sort = data.sort_values(['label'])
y = data_sort.label.values
id = data_sort.id.values

# se intenta convir1tiendo las imagenes a grises
X_list = []
for j, i in enumerate(id):
    sys.stdout.write("lectura de img: %d%%   \r" % (j*100//len(id)))
    sys.stdout.flush()
    a = cv2.imread('./train/{}.tif'.format(id[0]))  # se lee las imagenes
    # b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)  # se cambia a escala de grises
    c = a[31:31+32, 31:31+32]  # se corta la imagen a 32 x 32
    # d = c.flatten()  # vconvierte un array 2d a 1d
    X_list.append(c)
print("lectura de img: %d%%   \r" % (j*100//len(id)+1))

X = np.array(X_list)
print("imagens a usar: "+str(X.shape)+"\n"+"labels = "+str(y.shape))
X = np.array(X/255.0, dtype='float16')  # se normaliza los datos
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3)
nc = 2
Y = to_categorical(y, num_classes=nc)

x_train, x_valid, y_train, y_valid = train_test_split(X, Y,
                                                      train_size=0.9,
                                                      random_state=0)
# x_ = []
# y_ = []
# for i in range(len(y_train)):
#     sys.stdout.write("rotacion de img: %d%%   \r" % (i*100//len(y_train)))
#     sys.stdout.flush()
#     aux = x_train[i]
#     x_.append(aux)
#     y_.append(y_train[i])
#     for k in range(3):
#         aux = np.rot90(aux)
#         x_.append(aux)
#         y_.append(y_train[i])
# sys.stdout.write("rotacion de img: %d%%  s \r" % (i*100//len(y_train)))
# x_train = np.array(x_)
# y_train = np.array(y_)

modelo = ajuste_modelo(32, 32, 3, nc)

# model = CancerNet.build(width = 96, height = 96, depth = 3, classes = 2)

# Edit:: Adagrad(lr=1e-2, decay= 1e-2/10) was used previous;y
modelo.compile(optimizer=Adam(lr=0.0001),
               loss='binary_crossentropy', metrics=['accuracy'])


filepath = "checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')  # Save Best Epoc

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]  # LR Scheduler Used here

history = modelo.fit(x_train, y_train, steps_per_epoch=len(y_train)//32,
                     validation_data=(x_valid, y_valid),
                     validation_steps=len(y_valid)//32,
                     epochs=11,
                     verbose=1,
                     callbacks=callbacks_list)


# epocas = 11
# tamanho_lote = len(y_train)//(32**2)
# modelo.fit(x_train, y_train, epochs=epocas, batch_size=tamanho_lote,
#            verbose=1, validation_data=(x_valid, y_valid))

model_jason = modelo.to_json()
with open("modelo.json", 'w') as j_f:
    j_f.write(model_jason)

modelo.save_weights("modelo.hi5")
print("guardado en el disco")
