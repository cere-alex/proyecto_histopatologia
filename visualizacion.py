#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import os
# se observara el numero de archivos a usar
train = pd.read_csv("./train_labels.csv")
print('lenght train_labels.csv = '+str(len(train)))
print("length ./train = " + str(len(os.listdir('./train'))))

test = pd.read_csv('./sample_submission.csv')
print('lenght sample_submission.csv = '+str(len(test)))
print("length ./test = " + str(len(os.listdir('./test'))))
# se observa que la longitud de los archivos son correctos

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

"""
En loa labels se hara un histograma para saber cuantos 0 y 1 tenemos
"""
plt.bar([0, 1], [(train['label'] == 0).sum(), (train['label'] == 1).sum()])
plt.show()

# se observa que hay mayor cantidad de imagenes donde no existe la patologia

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
fig = plt.figure(figsize=(nro, nro//2), dpi=150)
for plotNr, i in enumerate(marco):
    ax = fig.add_subplot(2, nro//2, plotNr+1, xticks=[], yticks=[])  # subplot
    plt.imshow(i)  # plot image
    ax.set_title('Label: ' + str(label_list[plotNr]))  # muestar los labels

fig.show()
