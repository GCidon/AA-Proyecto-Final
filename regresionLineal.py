import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from scipy.optimize import fmin_tnc

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

def calcOptTheta(X, Y, maxIt, landa):
    result = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(polyX, Y, 1))
    return result[0]

def evRegresionLogistica(X, Y, t):
    aciertos = 0
    res = 0 
    totales = len(Y)
    aux = 0
    
    for i in X:
        if sigmoide(np.dot(i, t)) >= 0.5:
            res = 1
        else:
            res = 0
        
        if Y[aux] == res:
            aciertos += 1

        aux += 1
            
    porcentaje = aciertos / totales * 100
    print(82,100, str(porcentaje) , "% de aciertos")

def oneVsAll(X, y, num_etiquetas, reg, maxIt, landa):

    ThetasMatriz = np.zeros((num_etiquetas, X.shape[1]))

    i = 0
    while i < num_etiquetas:

        os.system('cls')
        print("Numero de etiquetas procesadas: ", i + 1, " de un total de ", num_etiquetas, " con lamda = ", landa, " y ", maxIt, " iteraciones.")
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


lamdas = [0.001, 0.01, 0.1, 1, 10, 50, 100, 300, 500]
maxIterations = [70, 100, 150, 200, 300]
num_labels = len(np.unique(yaux))

testedThetasValues = dict()
testedThetas = []
p = 0

aciertos = []
myLandas = []
myIter = []

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

os.system('cls')

val =  aciertos.index(max(aciertos)) #Lo mismo que max(testedThetasValues)

print("Mejor porcentaje de acierto para entrenamiento: " ,str(aciertos[val]) + "% de acierto con un valor de lambda = ", myLandas[val], " con ", myIter[val], " iteraciones.")

saveOutputData(myLandas, myIter, aciertos, testedThetas)
