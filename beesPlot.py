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

# Las imagenes no tan buenas
bee_test = os.listdir(carpeta+"/bee2")

resizedbees = []
etiqueta = [] # 0 para abeja, 1 para avispa

# Generar imagenes resizeadas
for i in bee_test:
    if os.path.isfile(carpeta + "/bee2/" + i):
        aux = Image.open(carpeta + "/bee2/" + i).convert('P')
        aux = aux.resize((100,100), Image.ANTIALIAS)
        aux = np.asarray(aux)/255.0
        resizedbees.append(aux)
        etiqueta.append(0)

imgbees = []
imgwasps = []

for i in range(1, len(resizedbees)):
    imgbees.append(resizedbees[i])

w=10
h=10
fig=plt.figure(figsize=(15,15))
columns = 4
rows = 5

for i in range(1,21):
    
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(resizedbees[random.randint(1,len(resizedbees))])
    converted_num = str(i) 
    plt.title("bee -"+converted_num)
    i=int(i)
    
plt.show()



