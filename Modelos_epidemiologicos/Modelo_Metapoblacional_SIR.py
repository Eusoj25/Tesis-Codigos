import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Definir los parámetros del modelo SIR metapoblación
n = 2  # Número de subpoblaciones
n_1 = 600 # La población del parche 1
n_2 = 400 # La población del parche 2
beta_11 = 0.001 # Tasa de infección en el parche 1
beta_21 = 0.001 # Tasa de infección del parche 1 al parche 2
beta_22 = 0.002 # Tasa de infección en el parche 2
beta_12 = 0.002 # Tasa de infección del parche 2 al parche 1
mu_1 = 0.00011 # Tasa de mortalidad en el parche 1
mu_2 = 0.00013 # Tasa de mortalidad en el parche 2
gamma_1 = 1/1.3 # Tasa de recuperación del parche 1
gamma_2 = 1/1.4 # Tasa de recuperación del parche 2

# Definir las matrices de la metapoblación
beta = np.array([[beta_11, beta_21], [beta_12, beta_22]])  # Matriz de las tasas de infección
mu = np.array([mu_1,mu_2])                                 # Vector de las tasas de mortalidad 
gamma = np.array([gamma_1,gamma_2])                        # Vector de las tasas de recuperación
N = np.array([n_1,n_2])                                    # Vector de las poblaciones 

# Número de reproducción para cada parche
R_1 = (beta_11*n_1)/(mu_1 + gamma_1)
R_2 = (beta_22*n_2)/(mu_2 + gamma_2)

#print(R_1, R_2)

# Definir las ecuaciones diferenciales del modelo SIR metapoblacional
def deriv(y, t, beta, mu, gamma, N):
    S = y[:n]     # Vector de susceptibles para cada subpoblación
    I = y[n:2*n]  # Vector de infectados para cada subpoblación

    # Tasa de cambio de S y I para cada subpoblación
    dS = np.zeros(n)
    dI = np.zeros(n)
    #print("---------S y I------------")
    #print(S)
    #print(I)

    # Calcular la tasa de cambio para cada subpoblación
    for i in range(n):
        dS[i] = mu[i]*N[i] - mu[i]*S[i]
        dI[i] = - (mu[i] + gamma[i])*I[i]
        for j in range(n):
             dS[i] -= S[i]*beta[i][j]*I[j]
             dI[i] += S[i]*beta[i][j]*I[j] 

    #print("---------dS y dI------------")
    #print(dS)
    #print(dI)         

    return np.concatenate((dS, dI))

# Condiciones iniciales de la simulación
S0 = [(n_1 - 7), (n_2 - 7)]   # Susceptibles iniciales en cada subpoblación
I0 = [7, 7]                   # Infectados iniciales en cada subpoblación
y0 = np.concatenate((S0, I0)) # Vector de condiciones iniciales para la simulación

# Vector de timesteps para la simulación
t = np.linspace(0, 20, 100)

# Resolver las ecuaciones diferenciales
sol = odeint(deriv, y0, t, args=(beta, mu, gamma, N))
print("---------Matriz Solución------------")
print(sol)

S = sol[:, :n]     # Vector de susceptibles para cada subpoblación
I = sol[:, n:2*n]  # Vector de infectados para cada subpoblación



# Graficar los resultados

plt.plot(t, S[:,0], label='S1')
plt.plot(t, S[:,1], label='S2')
plt.plot(t, I[:,0], label='I1')
plt.plot(t, I[:,1], label='I2')
plt.legend()
plt.show()
