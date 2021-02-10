import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as fmin_tnc
from scipy.io import loadmat
from sklearn.svm import SVC

def calcula_porcentaje(Y, Z):

    m = Y.shape[0]

    res = (Y==Z)
    aciertos = np.sum(res)

    return round((aciertos / m) * 100, 5)

data = loadmat('dataMat.mat')
X = data['X']
y = data['y'].ravel()
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']


svms = []
mejorsvm = 0

c = 0
sigma = 0

porcentaje = 0

cont = 0

vals = np.array([0.01, 0.1, 1, 10, 30])

for i in range(len(vals)):
    for j in range(len(vals)):
        svms.append(SVC(kernel='rbf', C=vals[i], gamma=1/(2*(vals[j]**2))))
        svms[cont].fit(X, y)

        h = svms[cont].predict(Xval)
        cosa = calcula_porcentaje(yval.ravel(), h)

        if(cosa > porcentaje):
            porcentaje = cosa
            c = vals[i]
            sigma = vals[j]
            mejorsvm = cont

        cont = cont+1

        print(cosa)
        print(cont)

svm = SVC(kernel='rbf', C=10, gamma=1/(2*(10**2)))
svm.fit(X, y)

h = svm.predict(Xtest)
resultado = calcula_porcentaje(ytest.ravel(), h)

print(resultado)