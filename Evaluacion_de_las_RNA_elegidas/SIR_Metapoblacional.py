import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SIR_Metapoblacional(object):

    def __init__(self, beta):
        """la matriz "beta" es una matriz 2x2, la cual contiene
        todas las tasas de infección de la metapoblación"""
        self.beta = beta
        self.n = 2
        self.n_1 = 600       # La población del parche 1
        self.n_2 = 400       # La población del parche 2
        self.mu_1 = 0.00011  # Tasa de mortalidad en el parche 1
        self.mu_2 = 0.00013  # Tasa de mortalidad en el parche 2
        self.gamma_1 = 1/1.3 # Tasa de recuperación del parche 1
        self.gamma_2 = 1/1.4 # Tasa de recuperación del parche 2


        # Definir las matrices de la metapoblación
        self.beta = np.array(beta)                         # Matriz de las tasas de infección
        self.mu = np.array([self.mu_1,self.mu_2])          # Vector de las tasas de mortalidad 
        self.gamma = np.array([self.gamma_1,self.gamma_2]) # Vector de las tasas de recuperación
        self.N = np.array([self.n_1,self.n_2])             # Vector de las poblaciones 
    
    # Simulación matemática del modelo epidemiológico con la ayuda de odeint
    def Simulate (self, S0, I0, t):

        # Definir las ecuaciones diferenciales del modelo SIR metapoblacional
        def deriv(y,t,beta,mu,gamma,N):
            S = y[:self.n]          # Vector de susceptibles para cada subpoblación
            I = y[self.n:2*self.n]  # Vector de infectados para cada subpoblación

            # Tasa de cambio de S y I para casa subpolación
            dS = np.zeros(self.n)
            dI = np.zeros(self.n)

            # Calcular la tasa de cambio para cada subpoblación
            for i in range(self.n):
                dS[i] = mu[i]*N[i] - mu[i]*S[i]
                dI[i] = - (mu[i] + gamma[i])*I[i]
                for j in range(self.n):
                    dS[i] -= S[i]*beta[i][j]*I[j]
                    dI[i] += S[i]*beta[i][j]*I[j]

            return np.concatenate((dS,dI))
        

        t = t
        y0 = np.concatenate((S0,I0)) # Vector de condiciones iniciales para la simulación

        # Resolver las ecuaciones diferenciales
        sol = odeint(deriv, y0, t, args=(self.beta,self.mu,self.gamma,self.N))

        return sol
        



        
        
            
    



    

    




