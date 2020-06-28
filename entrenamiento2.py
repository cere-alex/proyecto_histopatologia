#!/usr/bin/env python3

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import SeparableConv2D, Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import shutil
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import cv2
import seaborn as sns
from random import shuffle


def Histopatologia(ancho, alto, profundidad, clases):
    """
    Se crea un posible modelo.
    """
    model = Sequential()
    chanDim = -1
    # CONV 1
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu',
                     input_shape=(alto, ancho, profundidad)))
    # model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # (CONV 2)
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    # model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    # softmax classifier
    model.add(Dense(clases, activation="softmax"))
    model.summary()

    # return the constructed network architecture
    return model


len(os.listdir('./test/'))
len(os.listdir('./train/'))

df = pd.read_csv('./train_labels.csv')
"""
Leyendo los avances en Kaggle se observo que estas dos imagenes
causan problemas en el entrenamiento

"""
df = df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

df_train = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# df_train.set_index(['id'], inplace=True)


"""
Las imagenes de train se dividiran en grupo de entranamiento y validadcion.
"""
df_train, df_valid = train_test_split(df_train,
                                      train_size=0.9,
                                      random_state=0)

y_train = df_train['label'].values
x_train_id = df_train['id'].values
y_valid = df_valid['label'].values
x_valid_id = df_valid['id'].values
"""
Se cortara la imagen central de 32x32 para el entranamiento
"""

if os.path.exists('X_train.npy'):
    X_train = np.load('X_train.npy')
    print("Se cargo el archivo X_train.npy")
else:
    X_train = []
    for j, i in enumerate(x_train_id):
        sys.stdout.write("lectura de df_train: %d%%\r" % (j*100//len(x_train_id)))
        sys.stdout.flush()
        a = cv2.imread('./train/{}.tif'.format(i))  # se lee las imagenes
        b = a[31:31+32, 31:31+32]  # se corta la imagen a 32 x 32 x 3
        X_train.append(b)
    print("lectura de df_train: %d%%   \r" % (j*100//len(x_train_id)+1))
    X_train = np.array(X_train)
    np.save('X_train.npy', X_train)

if os.path.exists('X_valid.npy'):
    X_valid = np.load('X_valid.npy')
    print("Se cargo el archivo X_valid.npy")
else:
    X_valid = []
    for j, i in enumerate(x_valid_id):
        sys.stdout.write("lectura de df_valid: %d%%\r" % (j*100//len(x_valid_id)))
        sys.stdout.flush()
        a = cv2.imread('./train/{}.tif'.format(i))  # se lee las imagenes
        b = a[31:31+32, 31:31+32]  # se corta la imagen a 32 x 32 x 3
        X_valid.append(b)
    print("lectura de df_valid: %d%%   \r" % (j*100//len(x_valid_id)+1))
    X_valid = np.array(X_valid)
    np.save('X_valid.npy', X_valid)
"""
SE ajustara los datos, y_train a categorical
"""

y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)
datagen = ImageDataGenerator(rescale=1.0/255)
datagen.fit(X_train)
X_y_train = datagen.flow(X_train, y_train, batch_size=32)
X_y_valid = datagen.flow(X_valid, y_valid, batch_size=32)

model = Histopatologia(32, 32, 3, 2)

model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
plot_model(model, to_file='model.png')

train_steps = np.ceil(len(y_train) / 32)
val_steps = np.ceil(len(y_valid) / 32)

filepath = "checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')  # Save Best Epoc

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]  # LR Scheduler Used here

history = model.fit_generator(X_y_train, steps_per_epoch=train_steps,
                              validation_data=X_y_valid,
                              validation_steps=val_steps,
                              epochs=11,
                              verbose=1,
                              callbacks=callbacks_list)
