import tensorflow as tf 
import numpy as np
import SIR_Metapoblacional as modelo
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

beta = np.array([[0.001, 0.001], [0.002, 0.002]])
model = modelo.SIR_Metapoblacional(beta)

# Lista para almacenar las matrices de tasas de infección
betas = []

# Lista para almacenar los objetos de la clase 'SIR_Metapoblacional'
models = []

# Crear objetos de la clase 'SIR_Metapoblacional' con matrices de tasas de infección incrementadas
while beta[0][0] <= 0.1 and beta[1][1] <= 0.1:
    beta[0][0] = beta[0][0] + 0.0001
    beta[1][1] = beta[1][1] + 0.0001
    beta[0][1] = beta[0][1] + 0.00005
    beta[1][0] = beta[1][0] + 0.00005
    betas.append(np.copy(beta))
    model = modelo.SIR_Metapoblacional(beta)
    models.append(model)


#print(betas[0])
#print(betas[-1])
#print(len(betas[-1]))
#print(len(betas))
#print(len(models))


# Condiciones iniciales de la simulación
S0 = [(600 - 7), (400 - 7)]   # Susceptibles iniciales en cada subpoblación
I0 = [7, 7]                   # Infectados iniciales en cada subpoblación
y0 = np.concatenate((S0, I0)) # Vector de condiciones iniciales para la simulación

# Vector de timesteps para la simulación
t = np.linspace(0, 20, 100)

# Lista para almacenar los vectores de infectados para cada elemento en models
Is = []

# Realizar simulación para cada uno de los elementos de la lista models
for i in range(len(models)):
    sol = models[i].Simulate(S0, I0, t)
    I = sol[:,2:]
    #I_1 = sol[:,2]
    #I_2 = sol[:,3]
    #Is.append(np.concatenate((I_1,I_2)))
    Is.append(I)

#print("Is------")
#print(Is[0])
#print(Is[-1])
#print(len(Is))
#print(len(Is[-1]))


betas = np.array(betas)
Is = np.array(Is)

print(betas.shape)
print(Is.shape)

# Guardar la base de datos
np.save('betas_2D.npy', betas)
np.save('Is_2D.npy', Is)





