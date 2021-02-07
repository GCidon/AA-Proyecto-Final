import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat

def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

def hMatrix(x, theta):
    return np.dot(x, theta.T)

def sigmoid(z): 
    return (1 / (1 + np.exp(-z)))

def regularizedCost(theta, lamda: float, X, Y):
    m = X.shape[0]

    theta=theta.reshape(-1, Y.shape[1])

    aux = (1/(2*m)) * np.sum((np.dot(X, theta) - Y)**2)
    regTerm = (lamda / (2 * m)) * np.sum(theta[1:len(theta)]**2) 

    return aux + regTerm


def regularizedGradient(theta, lamda: float, X, Y):
    m = X.shape[0]

    theta=theta.reshape(-1, Y.shape[1])

    grad = (1/ m) * np.dot(X.T, np.dot(X, theta) - Y)

    regTerm = grad + (lamda/m)*theta

    regTerm[0] = grad[0]
    return regTerm.flatten()

def regularizedBoth(theta, lamda, X, Y):
    cost = regularizedCost(theta, lamda, X, Y)
    grad = regularizedGradient(theta, lamda, X, Y)
    return cost, grad

def calcOptTheta(lamda, X, Y):

    theta = np.zeros([X.shape[1], 1])

    def optAux(theta):
        return regularizedBoth(theta, lamda, X, Y)

    theta = opt.minimize(fun=optAux, x0=theta, method='CG', jac=True, options={'maxiter':200})
    return theta.x    

def trainingCurva(X, Y, Xval, Yval, lamda):
    m = X.shape[0]

    auxTrain = np.zeros(m)
    auxVal = np.zeros(m)

    for i in range(0,m):
        j = i+1
        theta = calcOptTheta(lamda, X[:j], Y[:j])
        auxTrain[i] = regularizedCost(theta, lamda, X[:j], Y[:j])
        auxVal[i] = regularizedCost(theta, lamda, Xval, Yval)

    return auxTrain, auxVal

def validationCurva(X, Y, Xval, Yval):
    m = X.shape[0]
    lamdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    auxTrain = np.zeros((len(lamdas), 1))
    auxVal = np.zeros((len(lamdas), 1))
    for i in range(len(lamdas)):
        lamda = lamdas[i]
        theta = calcOptTheta(lamda, X, Y)
        auxTrain[i] = regularizedCost(theta, 0, X, Y)
        auxVal[i] = regularizedCost(theta, 0, Xval, Yval)

    return auxTrain, auxVal, lamdas

def newDatos(X, pow):
    ret = X
    for i in range(1, pow):
        ret = np.column_stack((ret, (X**(i+1))))
    return ret

def normalizar(X):
    media = np.mean(X, axis=0)
    desviacion = np.std(X, axis=0)
    X = (X- media) / desviacion
    return X, media, desviacion

data = loadmat('ex5data1.mat')
X = data['X'] 
y = data['y']

Xval = data['Xval']
yval = data['yval']

Xtest = data['Xtest']
ytest = data['ytest']

m = X.shape[0]
Xones = np.hstack([np.ones([m, 1]), X])

m2 = Xval.shape[0]
Xvalones = np.hstack([np.ones([m2, 1]), Xval])

##############################################################

# thetas = np.array([[1], [1]])

# print(regularizedCost(thetas, 1, Xones, y))
# print(regularizedGradient(thetas, 1, Xones, y))

# thetas = calcOptTheta(0, Xones, y)

# plt.plot(X, y, 'x')
# plt.plot(X, np.dot(np.insert(X, 0, 1, axis=1), thetas))
# plt.show()

##############################################################

# auxTrain, auxVal = trainingCurva(Xones, y, Xvalones, yval, 0)

# plt.plot(range(0,m), auxTrain)
# plt.plot(range(0,m), auxVal)
# plt.show()

##############################################################

# X2 = newDatos(X, 8)
# Xnormal, media, desviacion = normalizar(X2)

# Xval2 = newDatos(Xval, 8)
# Xvalnormal = (Xval2 - media) / desviacion

# m3 = Xnormal.shape[0]
# Xnormalones = np.hstack((np.ones((m3, 1)), Xnormal))

# m4 = Xvalnormal.shape[0]
# Xvalnormalones = np.hstack((np.ones((m4, 1)), Xvalnormal))

# thetas = calcOptTheta(0, Xnormalones, y)

# plt.plot(X, y, 'x')

# auxX = np.array(np.arange(min(X)-15, max(X)+25, 0.05))

# Xpol = (newDatos(auxX, 8) - media) / desviacion

# Xpol = np.insert(Xpol, 0, 1, axis=1)
# plt.plot(auxX, np.dot(Xpol, thetas), '-')

# plt.show()

# auxTrain, auxVal = trainingCurva(Xnormalones, y, Xvalnormalones, yval, 0)

# plt.plot(range(0,m), auxTrain)
# plt.plot(range(0,m), auxVal)
# plt.show()

##############################################################

X2 = newDatos(X, 15)
Xnormal, media, desviacion = normalizar(X2)

Xval2 = newDatos(Xval, 15)
Xvalnormal = (Xval2 - media) / desviacion

Xtest2 = newDatos(Xtest, 15)
Xtestnormal = (Xtest2 - media) / desviacion

m3 = Xnormal.shape[0]
Xnormalones = np.hstack([np.ones([m3, 1]), Xnormal])

m4 = Xvalnormal.shape[0]
Xvalnormalones = np.hstack([np.ones([m4, 1]), Xvalnormal])

m5 = Xtestnormal.shape[0]
Xtestnormalones = np.hstack([np.ones([m5, 1]), Xtestnormal])

auxTrain, auxVal, lamdas = validationCurva(Xnormalones, y, Xvalnormalones, yval)

plt.plot(lamdas, auxTrain)
plt.plot(lamdas, auxVal)
plt.show()

thetas = calcOptTheta(3, Xnormalones, y)

coste = regularizedCost(thetas, 0, Xtestnormalones, ytest)

print(coste)
