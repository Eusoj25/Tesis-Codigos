import tensorflow as tf 

import datetime
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten 
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb
from wandb.keras import WandbCallback

# Cargar las matrices de infección interpoblacional y las curvas de infección
betas = np.load(r'Datos\betas.npy')
Is = np.load(r'Datos\Is.npy')

#print(betas.shape)
#print(Is.shape)

# Dividir los datos en datos de entrenamiento y validación
x_train, x_test, y_train, y_test = train_test_split(Is, betas, test_size=0.2, random_state=1)

y_trainv = y_train.reshape(784,4)
y_testv = y_test.reshape(196,4)

#print(x_train.shape)

# crea un objeto StandardScaler
scaler = StandardScaler()

# ajusta el scaler en los datos de entrenamiento
scaler.fit(x_train)

# transforma los datos de entrenamiento y de prueba con el scaler ajustado
x_trainv = scaler.transform(x_train)
x_testv = scaler.transform(x_test)

# Le cambiamos la forma a las x, de (200,1) a (100,2)
x_trainv = x_trainv.reshape((784,100, 2), order='F')
x_testv = x_testv.reshape((196,100, 2), order='F')

# Crear un modelo secuencial 
model = Sequential()
model.add(Conv2D(10,2, input_shape=(100,2,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,1)))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(4))
model.add(Activation('sigmoid'))

model.summary()

# Parámetros del modelo
learning_rate = 0.0002
epochs = 50
batch_size = 20
metric = 'mape'
loss = 'mse'

# Configurar w&b
wandb.init(project='The best models')
wandb.config.learning_rate = learning_rate
wandb.config.batch_size = batch_size
wandb.config.epochs = epochs
wandb.config.metric = metric
wandb.config.loss = loss
wandb.config.optimizer = 'rmsprop'

# Configurar el modelo
model.compile(loss=loss,optimizer=RMSprop(learning_rate=learning_rate),metrics=metric)

# Crear una carpeta para guardar los datos para tensorboard
log_dir="Graph/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Configurar tensorboard
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/

model.fit(x_trainv, y_trainv,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(x_testv, y_testv),
         callbacks= [tbCallBack, WandbCallback()])

score = model.evaluate(x_testv, y_testv, verbose=1)
print(score)

# Evaluar el modelo
a = model.predict(x_testv)
print("Lo que obtuvo la red fue: ")
print(a[25])
print("El resultado correcto: ")
print(y_testv[25])

wandb.config.predic = a[25]
wandb.config.correct = y_testv[25]

# Guardar el modelo en disco
model.save("The best Red Conv2D.h5")
