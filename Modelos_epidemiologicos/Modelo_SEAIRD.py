import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Definir las condiciones iniciales 
N = 261         # Población total
E_0 = 5         # Número inicial de individuos expuestos
A_0 = 0         # Número inicial de individuos asintomáticos 
I_0 = 0         # Número inicial de infectados
R_0 = 0         # Número inicial de Recuperados
D_0 = 0         # Número inicial de personas muertas por la enfermedad 
S_0 = N - E_0   # Número inicial de susceptibles

# Definir los parámetros del modelo
eta = 0.2      # Reducción de riesgo de infección debido al aislamiento
sigma = 0.265      # Tasa de transición de expuestos a infecciosos
m = 0.5         # Fracción de expuestos que evolucionan a asintomático
gamma = 0.5      # 1/gamma es el tiempo característico de la etapa infecciosa
mu = 0.02         # La fracción de enfermos que mueren por la enfermedad
beta = (gamma*3.25)/(eta*(1.-m) + m + (gamma/sigma))    # Riesgo de infección 
t = np.linspace(0, 30, 100)   # Tiempo de simulación

# Definimos el número de reproductivo básico
R_0 = beta*(1/sigma + (m + eta*(1-m))/gamma)
print(R_0)

# Definir el modelo matemático
def SEAIRD_model(y, t, beta, eta, sigma, m, gamma, mu):
    S, E, A, I, R, D = y
    dS = - beta*S*(E/N) - beta*S*(A/N) - beta*eta*S*(I/N)
    dE = beta*S*(E/N) + beta*S*(A/N) + beta*eta*S*(I/N) - gamma*E
    dA = m*sigma*E - gamma*A
    dI = (1 - m)*sigma*E - gamma*I
    dR = gamma*(A + (1 - mu)*I)
    dD = gamma*mu*I
    return dS, dE, dA, dI, dR, dD

# Configurar la simulación
y0 = S_0, E_0, A_0, I_0, R_0, D_0 
ret = odeint(SEAIRD_model, y0, t, args=(beta, eta, sigma, m, gamma, mu))
S, E, A, I, R, D = ret.T

# Visualizar los resultados
plt.plot(t, S, label='Susceptibles')
plt.plot(t, E, label = 'Expuestos')
plt.plot(t, A, label= 'Asintomáticos')
plt.plot(t, I, label='Infectados')
plt.plot(t, R, label = 'Recuperados')
plt.plot(t, D, label = 'Muertos')
plt.legend()
plt.show()

