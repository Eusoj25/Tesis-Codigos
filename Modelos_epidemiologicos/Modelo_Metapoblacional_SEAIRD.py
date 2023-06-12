import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Definir los parámetros del modelo SEAIRD metapoblación
n = 2  # Número de subpoblaciones
n_1 = 1200 # La población del parche 1
n_2 = 800 # La población del parche 2
N = np.array([n_1,n_2]) # Vector de las poblaciones 
eta = 0.2      # Reducción de riesgo de infección debido al aislamiento
sigma = 0.265      # Tasa de transición de expuestos a infecciosos
m = 0.5         # Fracción de expuestos que evolucionan a asintomático
gamma = 0.5      # 1/gamma es el tiempo característico de la etapa infecciosa
mu = 0.02         # La fracción de enfermos que mueren por la enfermedad
beta = (gamma*3.25)/(eta*(1.- m) + m + (gamma/sigma))    # Riesgo de infección 

# Definamos los parametros de la matriz de tiempo de residencia P
p_11 = 0.04
p_12 = 0.02 # fracción de la población de la zona 1 que en promedio se encuentra en cualquier instante en la zona 2.
p_21 = 0.06
p_22 = 0.08

# Definamos la matriz de tiempo de residencia
P = np.array([[p_11, p_12], [p_21, p_22]])

# Definamos el número promedio de individuos que se encuentran en cualquier momento en la región i
w_1 = sum([P[j][0] * N[j] for j in range(n)])
w_2 = sum([P[j][1] * N[j] for j in range(n)])
w =  np.array([w_1,w_2])

# Definir las ecuaciones diferenciales del modelo SEAIRD metapoblacional
def deriv(y, t, N, beta, eta, sigma, m, gamma, mu, P, w):
    S = y[:n]       # Vector de susceptibles para cada subpoblación
    E = y[n:2*n]    # Vector de Expuestos para cada subpoblación
    A = y[2*n:3*n]  # Vector de Asintomáticos para cada subpoblación
    I = y[3*n:4*n]  # Vector de infectados para cada subpoblación
    R = y[4*n:5*n]  # Vector de Recuperados para cada subpoblación
    D = y[5*n:6*n]  # Vector de Muertos por la enfermedad para cada subpoblación
    
    # Total de individuos expuestos en la zona k
    epsilon_1 = sum([P[j][0]*E[j] for j in range(n)])
    epsilon_2 = sum([P[j][1]*E[j] for j in range(n)])
    epsilon =  np.array([epsilon_1, epsilon_2])

    # Total de asintomáticos en la zona k
    At_1 = sum([P[j][0]*A[j] for j in range(n)])
    At_2 = sum([P[j][1]*A[j] for j in range(n)])
    At = np.array([At_1, At_2])

    # Total de infectados en zona k 
    It_1 = sum([P[j][0]*I[j] for j in range(n)])
    It_2 = sum([P[j][1]*I[j] for j in range(n)])
    It = np.array([It_1, It_2])

    # Tasa de cambio de S, E, A, I, R y D para cada subpoblación
    dS = np.zeros(n)
    dE = np.zeros(n)
    dA = np.zeros(n)
    dI = np.zeros(n)
    dR = np.zeros(n)
    dD = np.zeros(n)

    # Calcular la tasa de cambio para cada subpoblación
    for i in range(n):
        dS[i] = 0
        dE[i] = - sigma*E[i]
        dA[i] = m*sigma*E[i] - gamma*I[i]
        dI[i] = (1. - m)*sigma*E[i] - gamma*I[i]
        dR[i] = gamma*(A[i] + (1. - mu)*I[i])
        dD[i] = gamma*mu*I[i]
        for k in range(n):
            dS[i] -= beta*S[i]*(P[i][k]/w[k])*(epsilon[k] + At[k] + eta*It[k])
            dE[i] += beta*S[i]*(P[i][k]/w[k])*(epsilon[k] + At[k] + eta*It[k])

    return np.concatenate((dS, dE, dA, dI, dR, dD))

# Condiciones iniciales de la simulación
S0 = [(n_1 - 10), (n_2 - 5)]  # Susceptibles iniciales en cada subpoblación
E0 = [10,5]                   
A0 = [0,0]
I0 = [0, 0]
R0 = [0,0]
D0 = [0,0]
y0 = np.concatenate((S0, E0, A0, I0, R0, D0)) # Vector de condiciones iniciales para la simulación

# Vector de timesteps para la simulación
t = np.linspace(0, 100, 1000)

# Resolver las ecuaciones diferenciales
ret = odeint(deriv, y0, t, args=(beta, eta, sigma, m, gamma, mu, N, P, w))
S = ret[:, :n]      # Vector de susceptibles para cada subpoblación
E = ret[:,n:2*n]
A = ret[:,2*n:3*n]
I = ret[:,3*n:4*n]
R = ret[:,4*n:5*n]
D = ret[:,5*n:6*n]

# Graficar los resultados
import matplotlib.pyplot as plt
plt.plot(t, S[:,0], label='S1')
plt.plot(t, S[:,1], label='S2')
plt.plot(t, E[:,0], label='E1')
plt.plot(t, E[:,1], label='E2')
plt.plot(t, A[:,0], label='A1')
plt.plot(t, A[:,1], label='A2')
plt.plot(t, I[:,0], label='I1')
plt.plot(t, I[:,1], label='I2')
plt.plot(t, R[:,0], label='R1')
plt.plot(t, R[:,1], label='R2')
plt.plot(t, D[:,0], label='D1')
plt.plot(t, D[:,1], label='D2')
plt.legend()
plt.show()












            
  
