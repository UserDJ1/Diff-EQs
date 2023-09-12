# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:27:22 2023

@author: theus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


plt.style.use('seaborn-poster')
%matplotlib inline

# Define ODE
#f = lambda t, s: np.exp(-t) 
f = lambda t, s: t + s/5 

# Define parameters
h = 0.5 # Step size
t = np.arange(0, 5+h, h) # Numerical grid
s0 = -3 # Initial Condition

# Numericall integrate ODE using off-the-shelf commands
sol = solve_ivp(f, t, [s0], method='DOP853', dense_output=True)
z = sol.sol(t)

#Euler method
s = np.zeros(len(t))
s[0] = s0

for i in range(0, len(t) - 1):
    s[i + 1] = s[i] + h*f(t[i], s[i])

# Plotting the results.
plt.figure(figsize = (12, 8))
plt.plot(t, s, 'bo--', label='Approximate')
plt.plot(t, z.T, 'g', label='Off the shelf command')
plt.title('Eulers method')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
plt.show()