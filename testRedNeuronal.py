import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients
from displayData import displayData
import scipy.io as out
import os

def forward_prop(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3

def calcAciertos(Y, h):
    aciertos = 0
    totales = len(Y)
    dimThetas = len(h)

    for i in range(dimThetas):
        r = np.argmax(h[i])
        if(r==Y[i]):
            aciertos+=1     

    porcentaje = aciertos / totales * 100
    return porcentaje

data = loadmat("dataMat.mat")

X = data["Xtest"]
y = data["ytest"].ravel()

pesos = loadmat("pesos.mat")

theta1 = pesos["theta1"]
theta2 = pesos["theta2"]

a1, z2, a2, z3, a3 = forward_prop(X, theta1, theta2)

res = calcAciertos(y, a3)

print(res)

