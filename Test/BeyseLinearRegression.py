# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])

def phi(x):
    s = 0.1
    return np.append(1, np.exp(-(x - np.arange(0, 1 + s, s)) ** 2 / (2 * s * s)))
    

alpha = 0.1
beta = 9.0
PHI = np.array([phi(x) for x in X])
Sigma_N = np.linalg.inv(alpha*np.identity(PHI.shape[1])+beta*np.dot(PHI.T,PHI))
mu_N = beta*np.dot(Sigma_N, np.dot(PHI.T,t))

w = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, t))

xlist = np.arange(0, 1, 0.01)
ylist = [np.dot(w, phi(x)) for x in xlist]

plt.plot(xlist, [np.dot(w, phi(x)) for x in xlist], 'g') 
plt.plot(xlist, [np.dot(mu_N, phi(x)) for x in xlist], 'b')
plt.plot(X, t, 'o')
plt.show()