#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random

train = pd.read_csv("./train_labels.csv")
'''
la variable nro muestra la cantidade de imagenes
que tomar en cuenta en cabezera y en cola
'''
nro = 10  # ingresar un nro par
cabezera = train.head(n=int(nro / 2))
cola = train.tail(n=int(nro / 2))

muestra = pd.concat([cabezera, cola])

# se observo que el tamaño de las fotos son de 96 x 96
# pero leyendo las observaciones en kaggle dice que
# solo tenemos que observar un cuadro central de 32x32

# para esto se dibujara un cuadro negro para en las imagenes de muestra


im_array = np.array([np.asarray(Image.open("./train/{}.tif".format(i)))
                     for i in muestra.id], dtype='uint8')


cuadro = np.ones([nro, 96, 96, 3])
cuadro[:, 31, 31:-32, :] = 0
cuadro[:, 31+32, 31:-32, :] = 0
cuadro[:, 31:-32, 31, :] = 0
cuadro[:, 31:-32, 31+32, :] = 0

marco = cuadro * im_array

marco = marco.astype('uint8')

label_list = muestra.label.to_list()
im_array.shape
for n, i in enumerate(marco):
    plt.imshow(i)
    plt.title(label_list[n])
    plt.show()
