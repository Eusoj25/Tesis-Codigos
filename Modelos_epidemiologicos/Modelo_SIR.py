import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Definir las variables y parámetros del modelo
N = 261      # Población total
I_0 = 7         # Número inicial de infectados
S_0 = N - I_0  # Número inicial de susceptibles
beta = 0.0178    # Tasa de infección
alpha = 2.73    # Tasa de recuperación
t = np.linspace(0, 5, 100)   # Tiempo de simulación

# Definir el modelo matemático
def SIR_model(y, t, beta, alpha):
    S, I = y

    dS = -beta * S * I 
    dI = (beta * S - alpha )* I

    return dS, dI

# Configurar la simulación
y0 = S_0, I_0
sol = odeint(SIR_model, y0, t, args=(beta, alpha))
S, I = sol.T

# Visualizar los resultados
plt.plot(t, S, label='Susceptibles')
plt.plot(t, I, label='Infectados')
plt.legend()
plt.show()


