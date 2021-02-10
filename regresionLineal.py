import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from scipy.optimize import fmin_tnc
import scipy.io as out


def sigmoide(z):
    return 1/(1+np.exp(-z))

def coste(theta, X, Y, lmbda):
    H = sigmoide(np.matmul(X, theta))

    cost = (-1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))

    ret = cost + np.sum(np.square(theta)) * lmbda / 2 * m

    return ret

def gradiente(theta, X, Y, lmbda):
    
    H = sigmoide(np.matmul(X, theta))

    grad = (1 / len(Y)) * np.matmul(X.T, H - Y)

    aux = np.r_[[0], theta[1:]]

    ret = grad +(lmbda*aux/m)

    return ret

def oneVsAll(X, y, num_etiquetas, reg, maxIt, landa):

    ThetasMatriz = np.zeros((num_etiquetas, X.shape[1]))

    i = 0
    while i < num_etiquetas:

        print("Numero de etiquetas procesadas: ", i + 1, " Etiquetas: ", num_etiquetas, " Lamda: ", landa, " y ", " Iteraciones: " ,maxIt)
        auxY = (y == i).astype(int)
        ThetasMatriz[i, :] = calcOptTheta(X, auxY, maxIt, landa)
        i += 1

    return ThetasMatriz
def calcOptTheta(X, Y, maxIt, landa):
    result = opt.minimize(
        fun=coste_y_gradiente, 
        x0=np.zeros(X.shape[1]), 
        args=(X, Y, landa), 
        method='TNC', 
        jac=True, 
        options={'maxiter': maxIt})

    return result.x


def coste_y_gradiente(x0, X, Y, landa):
    return coste(x0,X,Y,landa), gradiente(x0, X, Y, landa)

def calcAciertos(X, Y, t):
    cont = 0
    aciertos = 0
    totales = len(Y)
    dimThetas = len(t)
    valores = np.zeros(dimThetas)

    for i in X:      
        p = 0
        for x in range(dimThetas):
            valores[p] = sigmoide(np.dot(i, t[x]))
            p+=1

        r = np.argmax(valores)

        if(r==Y[cont]):
            aciertos+=1     

        cont += 1

    porcentaje = aciertos / totales * 100
    return porcentaje



data = loadmat('dataMat.mat')
X = data['X'] 
y = data['y']

yaux = np.ravel(y) 

Xval = data['Xval']
yval = data['yval']

Xtest = data['Xtest']
ytest = data['ytest']

m = X.shape[0]
Xones = np.hstack([np.ones([m, 1]), X])

m2 = Xval.shape[0]
Xvalones = np.hstack([np.ones([m2, 1]), Xval])

testedThetasValues = dict()
testedThetas = []
p = 0

aciertos = []
myLandas = []
myIter = []

lamdas = [0.001, 0.01, 0.1, 1, 10, 50, 100, 250, 500]
maxIterations = [50, 100, 150, 200, 250, 300]
num_labels = len(np.unique(yaux))

for i in lamdas:
    for r in maxIterations:
        landa = i
        one = (oneVsAll(X, yaux, num_labels, i, r, landa))
        testedThetasValues[p] = np.mean(one)
        testedThetas.append(one)

        myLandas.append(i)
        myIter.append(r)
        aciertos.append(calcAciertos(X, yaux, one))
        p += 1

val =  aciertos.index(max(aciertos)) #Lo mismo que max(testedThetasValues)

pinta_frontera_recta(X,y,testedThetas)

dict1 = {
    "thetas" : testedThetas
}

out.savemat("regresionMatTrained.mat", dict1)

print("Mejor porcentaje de acierto para entrenamiento: " ,str(aciertos[val]) + "% de acierto. Lambda = ", myLandas[val], " iteraciones", myIter[val])

