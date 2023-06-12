import numpy as np
import SIR_Metapoblacional as modelo
import matplotlib.pyplot as plt

beta = np.array([[0.001, 0.001], [0.002, 0.002]])
model = modelo.SIR_Metapoblacional(beta)

# Condiciones iniciales de la simulación
S0 = [(600 - 7), (400 - 7)]   # Susceptibles iniciales en cada subpoblación
I0 = [7, 7]                   # Infectados iniciales en cada subpoblación
y0 = np.concatenate((S0, I0)) # Vector de condiciones iniciales para la simulación

# Vector de timesteps para la simulación
t = np.linspace(0, 20, 100)

sol = model.Simulate(S0,I0,t)
print(sol)

S = sol[:, :2]          # Vector de susceptibles para cada subpoblación
I = sol[:, 2:2*2]       # Vector de infectados para cada subpoblación

  
fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0, 0].plot(t, S[:,0], label='S1')
axs[0, 0].set_title('S1')
axs[0, 0].legend()

axs[0, 1].plot(t, S[:,1], label='S2')
axs[0, 1].set_title('S2')
axs[0, 1].legend()
    
axs[1, 0].plot(t, I[:,0], label='I1')
axs[1, 0].set_title('I1')
axs[1, 0].legend()
    
axs[1, 1].plot(t, I[:,1], label='I2')
axs[1, 1].set_title('I2')
axs[1, 1].legend()
    
plt.show()

