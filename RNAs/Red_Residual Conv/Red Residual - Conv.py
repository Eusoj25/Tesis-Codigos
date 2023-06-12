import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow import keras
from keras import regularizers
from keras.initializers import glorot_uniform
import tensorflow.keras.layers as layers
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

# Crear un modelo Residual
inputs = keras.Input(shape=(100,2,1))
x = layers.Conv2D(10,2, activation = 'relu', padding = "same")(inputs)
x = layers.Conv2D(20,2, activation = 'relu', padding = "same")(x)
x = layers.MaxPooling2D(2, padding = "same")(x)
residual = x

x = layers.Conv2D(10,2, activation = 'relu', padding = "same")(x)
x = layers.Conv2D(20,2, activation = 'relu', padding = "same")(x)
x = layers.MaxPooling2D(2, padding = "same")(x)
residual = layers.Conv2D(20,1, strides=2)(residual)
x = layers.add([x,residual])
residual = x

x = layers.Conv2D(10,2, activation = 'relu', padding = "same")(x)
x = layers.Conv2D(20,2, activation = 'relu', padding = "same")(x)
x = layers.MaxPooling2D(2, padding = "same")(x)
residual = layers.Conv2D(20,1, strides=2)(residual)
x = layers.add([x,residual])

x = layers.Flatten()(x)
x = layers.Dense(100, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(4, activation = 'sigmoid')(x)

model = keras.Model(inputs = inputs, outputs = outputs)

model.summary()

# Parámetros del modelo
learning_rate = 0.0005
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
wandb.config.initializer = 'NO'
wandb.config.regularizer = 'Dropout'

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
print(a[43])
print("El resultado correcto: ")
print(y_testv[43])

wandb.config.predic = a[43]
wandb.config.correct = y_testv[43]

# Guardar el modelo en disco
model.save("The best Red Residual Conv2D.h5")