#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

data = pd.read_csv('./train_labels.csv')
id = data.id.values
y = data.label.values
# se intenta convirtiendo las imagenes a grises
X = []
for i in id:
    a = cv2.imread('./train/{}.tif'.format(id[0]))  # se lee las imagenes
    b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)  # se cambia a escala de grises
    c = b[31:31+32, 31:31+32]  # se corta la imagen a 32 x 32
    d = c.flatten()  # vconvierte un array 2d a 1d
    X.append(d)


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)
