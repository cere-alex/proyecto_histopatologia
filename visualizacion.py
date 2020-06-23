# !/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

train = pd.read_csv("./train_labels.csv")
'''
la variable nro muestra la cantidade de imagenes
que tomar en cuenta en cabezera y en cola
'''
nro = 20  # ingresar un nro par
cabezera = train.head(n=int(nro / 2))
cola = train.tail(n=int(nro / 2))

muestra = pd.concat([cabezera, cola])

im_array = np.array([np.asarray(Image.open("./train/{}.tif".format(i)))
                     for i in muestra.id], dtype='uint8')

label_list = muestra.label.to_list()
im_array.shape
for n, i in enumerate(im_array):
    plt.imshow(i)
    plt.title(label_list[n])
    plt.show()
