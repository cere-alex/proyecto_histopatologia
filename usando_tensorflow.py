#!/usr/bin/env python3

from IPython import display
import time
from tensorflow.keras import layers
import PIL
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import tensorflow as tf
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images.shape
train_labels.shape
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def build(width, height, depth, classes):

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


def ajuste_modelo(alto, ancho, dimensiones, nro_clases):

    model = Sequential()
    chanDim = -1
    # primera convolucion
    model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same',
                              activation='relu',
                              input_shape=(alto, ancho, dimensiones)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # segunda convolucion

    model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same',
                              activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same',
                              activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(84, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(nro_clases, activation='softmax'))

    model.summary()
    adam = Adam(lr=0.0001)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model
