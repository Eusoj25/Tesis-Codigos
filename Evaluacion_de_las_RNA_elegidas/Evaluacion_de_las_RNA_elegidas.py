import tensorflow as tf 
import numpy as np
import SIR_Metapoblacional as modelo 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datos de x_train
x_train = np.load('x_train.npy')

#beta = np.array([[0.0689, 0.03495], [0.03595, 0.0699]])
beta1 = np.array([[0.0035, 0.0025], [0.0035, 0.0045]])
beta2 = np.array([[0.0915, 0.0465], [0.0475, 0.0925]])
beta3 = np.array([[0.0595, 0.0305], [0.0315, 0.0605]])
beta4 = np.array([[0.001, 0.0053], [0.0063, 0.011]])
beta5 = np.array([[0.0089, 0.0052], [0.0062, 0.0099]])

# Listas de las tasas de infección y de los objetos de la clase 'SIR_Metapoblacional'
betas = [beta1,beta2,beta3,beta4,beta5]
models = []

for i in range(len(betas)):
    model = modelo.SIR_Metapoblacional(betas[i])
    models.append(model)

#print(len(models))

# Condiciones iniciales de la simulación
S0 = [(600 - 7), (400 - 7)]   # Susceptibles iniciales en cada subpoblación
I0 = [7, 7]                   # Infectados iniciales en cada subpoblación
y0 = np.concatenate((S0, I0)) # Vector de condiciones iniciales para la simulación

# Vector de timesteps para la simulación
t = np.linspace(0, 20, 100)

# Lista para almacenar los vectores de infectados para cada elemento en models
Is = []
Is1 = []

# Realizar simulación para cada uno de los elementos de la lista models
for i in range(len(models)):
    sol = models[i].Simulate(S0, I0, t)
    I = sol[:,2:]
    I_1 = sol[:,2]
    I_2 = sol[:,3]
    Is.append(np.concatenate((I_1,I_2)))
    Is1.append(I)

#print(len(Is))
#print(Is[0])

betas = np.array(betas)
Is = np.array(Is)
Is1 = np.array(Is1)

betasv = betas.reshape(5,4)

#print(betasv[0])

# crea un objeto StandardScaler
scaler = StandardScaler()

# ajusta el scaler en los datos de entrenamiento
scaler.fit(x_train)

# transforma los datos de validación con el scaler ajustado
Isv = scaler.transform(Is)

#print(Is[0])
#print(betasv[0])
#print(Isv[0])

# Le cambiamos la forma a las Isv para los modelos Conv2D y ResNet Conv2D
IsConv = Isv.reshape((5,100, 2), order='F')

# Se cargan cada uno de los modelos a evaluar:
evaluar_model_R_Densa = tf.keras.models.load_model('The best Red Densa.h5')
evaluar_model_R_Conv1D = tf.keras.models.load_model('The best Red Conv1D.h5')
evaluar_model_R_Conv2D = tf.keras.models.load_model('The best Red Conv2D.h5')
evaluar_model_R_Residual_D = tf.keras.models.load_model('The best Red Residual Densa.h5')
evaluar_model_R_Residual_C = tf.keras.models.load_model('The best Red Residual Conv2D.h5')

# Se evaluan cada uno de los modelos
score_R_Densa = evaluar_model_R_Densa.evaluate(Isv,betasv,verbose = 1)
score_R_Conv1D = evaluar_model_R_Conv1D.evaluate(Isv,betasv,verbose = 1)
score_R_Conv2D = evaluar_model_R_Conv2D.evaluate(IsConv,betasv,verbose = 1)
score_R_Residual_D = evaluar_model_R_Residual_D.evaluate(Isv,betasv,verbose = 1)
score_R_Residual_C = evaluar_model_R_Residual_C.evaluate(IsConv,betasv,verbose = 1)

# Predicción realizada por cada modelo
pred1 = evaluar_model_R_Densa.predict(Isv)
pred1 = np.array(pred1)
pred1 = np.round(pred1, decimals=5)

pred2 = evaluar_model_R_Conv1D.predict(Isv)
pred2 = np.array(pred2)
pred2 = np.round(pred2, decimals=5)

pred3 = evaluar_model_R_Conv2D.predict(IsConv)
pred3 = np.array(pred3)
pred3 = np.round(pred3, decimals=5)

pred4 = evaluar_model_R_Residual_D.predict(Isv)
pred4 = np.array(pred4)
pred4 = np.round(pred4, decimals=5)

pred5 = evaluar_model_R_Residual_C.predict(IsConv)
pred5 = np.array(pred5)
pred5 = np.round(pred5, decimals=5)

# Configurar opciones de impresión
np.set_printoptions(suppress=True)

# Se imprimen las matrices de infección reales
print("Matrices de infección reales: ")
print(betasv[:])
print("\n")

# Se imprime el score y predicción de cada modelo
print("Lo que obtuvo la Red Densa fue:")
print("\n")
print('Test loss:', score_R_Densa[0]) 
print('Test mape:', score_R_Densa[1])
print("Predicciones obtenidas:")
print(pred1[:])

print("\n")

print("Lo que obtuvo la Red Conv1D fue:")
print("\n")
print('Test loss:', score_R_Conv1D[0]) 
print('Test mape:', score_R_Conv1D[1])
print("Predicciones obtenidas:")
print(pred2[:])

print("\n")

print("Lo que obtuvo la Red Conv2D fue:")
print("\n")
print('Test loss:', score_R_Conv2D[0]) 
print('Test mape:', score_R_Conv2D[1])
print("Predicciones obtenidas:")
print(pred3[:])

print("\n")

print("Lo que obtuvo la ResNet Densa fue:")
print("\n")
print('Test loss:', score_R_Residual_D[0]) 
print('Test mape:', score_R_Residual_D[1])
print("Predicciones obtenidas:")
print(pred4[:])

print("\n")

print("Lo que obtuvo la ResNet Conv2D fue:")
print("\n")
print('Test loss:', score_R_Residual_C[0]) 
print('Test mape:', score_R_Residual_C[1])
print("Predicciones obtenidas:")
print(pred5[:])

print("\n")
print(betasv[0])

# Lista de las predicciones 
betas_pred = [pred1[-1],pred2[-1],pred3[-1],pred4[-1],pred5[-1]]
betas_pred = np.array(betas_pred)
betas_pred = betas_pred.reshape(5,2,2)
pred_models = []

for i in range(len(betas_pred)):
    model = modelo.SIR_Metapoblacional(betas_pred[i])
    pred_models.append(model)

pred_Is = []

# Realizar simulación para cada uno de los elementos de la lista pred_models
for i in range(len(pred_models)):
    sol = pred_models[i].Simulate(S0, I0, t)
    I = sol[:,2:]
    #I_1 = sol[:,2]
    #I_2 = sol[:,3]
    #Is.append(np.concatenate((I_1,I_2)))
    pred_Is.append(I)

# Gráficar de cada una de las predicciones junto con la curva de infección real

fig, axs = plt.subplots(1, 2, figsize=(10,10))

axs[0].plot(t, Is1[-1][:,0], label='S1_real')
axs[0].plot(t, pred_Is[0][:,0], label='S1_RedDensa')
axs[0].plot(t, pred_Is[1][:,0], label='S1_RedConv1D')
axs[0].plot(t, pred_Is[2][:,0], label='S1_RedConv2D')
axs[0].plot(t, pred_Is[3][:,0], label='S1_ResNetResidual')
axs[0].plot(t, pred_Is[4][:,0], label='S1_ResNetConv2D')
axs[0].set_title('S1')
axs[0].legend()

axs[1].plot(t, Is1[-1][:,1], label='S2_real')
axs[1].plot(t, pred_Is[0][:,1], label='S2_RedDensa')
axs[1].plot(t, pred_Is[1][:,1], label='S2_RedConv1D')
axs[1].plot(t, pred_Is[2][:,1], label='S2_RedConv2D')
axs[1].plot(t, pred_Is[3][:,1], label='S2_ResNetResidual')
axs[1].plot(t, pred_Is[4][:,1], label='S2_ResNetConv2D')
axs[1].set_title('S2')
axs[1].legend()

plt.show()
