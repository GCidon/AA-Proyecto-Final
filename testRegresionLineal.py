import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from scipy.optimize import fmin_tnc
import scipy.io as out
import os

def sigmoide(z):
    return 1/(1+np.exp(-z))


def porcentajeAciertos(X, Y, t):
    cont = 0
    aciertos = 0
    totales = len(Y)
    valores = np.zeros(len(t))

    for i in X:      
        p = 0
        for x in range(len(t)):
            valores[p] = sigmoide(np.dot(i, t[x]))
            p+=1

        r = np.argmax(valores)

        if(r==Y[cont]):
            aciertos+=1     

        cont += 1

    porcentaje = aciertos / totales * 100
    return porcentaje


dataTest = loadmat('dataMat2BW.mat')
dataReg = loadmat('regresionMatTrainedCompleteBW.mat')

Xval = dataTest['Xval']
yval = dataTest['yval']
yaux = np.ravel(yval)

Xtest = dataTest['Xtest']
ytest = dataTest['ytest']
ytest = np.ravel(ytest)

m2 = Xval.shape[0]
Xvalones = np.hstack([np.ones([m2, 1]), Xval])

thetas = dataReg["thetas"]

aciertos = []
for i in thetas:
    perc = porcentajeAciertos(Xval, yaux, i)
    aciertos.append(perc)
    print(perc)

maxVal =  aciertos.index(max(aciertos))
print("Mejor porcentaje de acierto con los datos de validacion: ", str(aciertos[maxVal]) + "%")

porcentaje = porcentajeAciertos(Xtest, ytest, thetas[maxVal])

print("Mejor porcentaje de acierto con los datos de test: ",str(porcentaje) + "%")