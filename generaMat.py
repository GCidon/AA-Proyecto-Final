import matplotlib.pyplot as plt
import matplotlib.image as implt
import seaborn as sns
import random
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import scipy.io as out
import time

carpeta = "kaggle_bee_vs_wasp"

# Las imagenes buenas
bee_train = os.listdir(carpeta+"/bee1")
wasp_train = os.listdir(carpeta+"/wasp1")

# Las imagenes no tan buenas
bee_test = os.listdir(carpeta+"/bee2")
wasp_test = os.listdir(carpeta+"/wasp2")

################################################## TRAIN
resizedbees = []
resizedwasps = []
etiqueta = [] # 0 para abeja, 1 para avispa

# Generar imagenes resizeadas
for i in bee_train:
    if os.path.isfile(carpeta + "/bee1/" + i):
        aux = Image.open(carpeta + "/bee1/" + i).convert('L')
        aux = aux.resize((100,100), Image.ANTIALIAS)
        aux = np.asarray(aux)/255.0
        resizedbees.append(aux)
        etiqueta.append(0)

for i in wasp_train:
    if os.path.isfile(carpeta + "/wasp1/" + i):
        aux = Image.open(carpeta + "/wasp1/" + i).convert('L')
        aux = aux.resize((100,100), Image.ANTIALIAS)
        aux = np.asarray(aux)/255.0
        resizedwasps.append(aux)
        etiqueta.append(1)

x_train = np.concatenate((resizedbees, resizedwasps), axis=0)
x_train_etiqueta = np.asarray(etiqueta)
x_train_etiqueta = x_train_etiqueta.reshape(x_train_etiqueta.shape[0], 1)

################################################## TEST
resizedbees = []
resizedwasps = []
etiqueta = [] # 0 para abeja, 1 para avispa

# Generar imagenes resizeadas
for i in bee_test:
    if os.path.isfile(carpeta + "/bee2/" + i):
        aux = Image.open(carpeta + "/bee2/" + i).convert('L')
        aux = aux.resize((100,100), Image.ANTIALIAS)
        aux = np.asarray(aux)/255.0
        resizedbees.append(aux)
        etiqueta.append(0)

for i in wasp_test:
    if os.path.isfile(carpeta + "/wasp2/" + i):
        aux = Image.open(carpeta + "/wasp2/" + i).convert('L')
        aux = aux.resize((100,100), Image.ANTIALIAS)
        aux = np.asarray(aux)/255.0
        resizedwasps.append(aux)
        etiqueta.append(1)

x_test = np.concatenate((resizedbees, resizedwasps), axis=0)
x_test_etiqueta = np.asarray(etiqueta)
x_test_etiqueta = x_test_etiqueta.reshape(x_test_etiqueta.shape[0], 1)


##################################################


X = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((x_train_etiqueta, x_test_etiqueta), axis=0)
X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])


# Separar en entrenamiento, test y validacion
Xtrain, Xtest, ytrain, ytest = train_test_split(np.asarray(X), np.asarray(y), test_size=0.2)
Xtrain, Xval, ytrain, yval = train_test_split(np.asarray(Xtrain), np.asarray(ytrain), test_size=0.25)


# Guardar en un .mat
matX = np.array(Xtrain)
maty = np.array(ytrain)

dicc = {
    "X": matX,
    "y": maty,
    "Xval": Xval,
    "yval": yval,
    "Xtest": Xtest,
    "ytest": ytest
}

out.savemat("dataMatBW.mat", dicc)



