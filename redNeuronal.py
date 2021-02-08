import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients
import scipy.io as out
import os

def sigmoid(z): 
    return (1 / (1 + np.exp(-z)))

def coste(X, Y, theta1, theta2):
    m = Y.shape[0]

    a1, z2, a2, z3, a3 = forward_prop(X, theta1, theta2)

    ret = np.sum(np.sum(-Y * np.log(a3) - (1-Y) * np.log(1-a3)))

    return ret/m

def costeReg(X, Y, m, theta1, theta2, K):
    return coste(X, Y, theta1, theta2) + ((K/(2*m)) * np.sum(np.square(theta1[:, 1:]))) + (np.sum(np.square(theta2[:, 1:])))

def forward_prop(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, K):
    m = X.shape[0]

    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas+1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1): ], (num_etiquetas, (num_ocultas+1)))

    delta1 = np.zeros_like(theta1)
    delta2 = np.zeros_like(theta2)

    a1, z2, a2, z3, a3 = forward_prop(X, theta1, theta2)

    coste = costeReg(X, y, m, theta1, theta2, K)

    for t in range(m):
        a1t = a1[t, :]
        a2t = a2[t, :]
        ht = a3[t, :]
        yt = y[t]

        d3t = ht - yt
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))

        delta1 = delta1 + np.dot(d2t[1:,np.newaxis], a1t[np.newaxis,:])
        delta2 = delta2 + np.dot(d3t[:,np.newaxis], a2t[np.newaxis,:])

    delta1 = delta1 / m
    delta2 = delta2 / m

    delta1[:, 1:] = delta1[:, 1:] + (K * theta1[:, 1:]) / m
    delta2[:, 1:] = delta2[:, 1:] + (K * theta2[:, 1:]) / m

    gradiente = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return coste, gradiente

def pesosAleatorios(tam1, tam2):
    aux = 0.12
    ret = np.random.uniform(low=-aux, high = aux, size=(tam2, tam1))
    ret = np.hstack((np.ones((ret.shape[0], 1)), ret))
    return ret

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

data = loadmat("/content/drive/MyDrive/AA In the house/Proyecto/dataMat2.mat")

X = data["X"]
y = data["y"].ravel()
X2 = data["Xval"]
y2 = data["yval"].ravel()

entradas = X.shape[1]
etiquetas = len(np.unique(y))
ocultas = [25, 50]
landas = [0.01, 0.1, 1, 10]
iteraciones = [50]

y_onehot = np.zeros((len(y), etiquetas))
for i in range(len(y)):
    y_onehot[i][y[i]] = 1

mejores_thetas = []
porcentaje = 0

for i in landas:
    for j in iteraciones:
        for t in ocultas:

            theta1 = pesosAleatorios(len(X[0]), t)
            theta2 = pesosAleatorios(t, etiquetas)

            thetas = [theta1, theta2]

            pesos = np.concatenate([thetas[i].ravel() for i,_ in enumerate(thetas)])
            print("Optimizando thetas")
            thetasguays = opt.minimize(fun=backprop, x0=pesos, args=(entradas, t, etiquetas, X, y_onehot, i), method='TNC', jac=True, options={'maxiter':j})
            print("Thetas optimizadas")
            theta1opt = np.reshape(thetasguays.x[:t * (entradas + 1)], (t, (entradas + 1)))
            theta2opt = np.reshape(thetasguays.x[t * (entradas + 1):], (etiquetas, (t + 1)))

            a1, x2, a2, z3, h = forward_prop(X2, theta1opt, theta2opt)

            res = calcAciertos(y2, h)

            if(res > porcentaje):
                porcentaje = res
                mejores_thetas = [theta1opt, theta2opt]

            print("Resultado con lambda:" + str(i) + ", iteraciones " + str(j) + ", capas ocultas " + str(t))
            print(res)

dicc = {
    "theta1" : mejores_thetas[0],
    "theta2" : mejores_thetas[1]
}

out.savemat("pesos.mat", dicc)



    
