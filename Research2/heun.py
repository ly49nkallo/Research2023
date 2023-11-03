import random
import numpy as np
import matplotlib.pyplot as plt

def Hopf_dots(mu, w0, alpha, beta, X, Y):
    X_dot = mu*X - w0*Y - (alpha*X - beta*Y)*(X**2 + Y**2)
    Y_dot = mu*Y + w0*X - (alpha*Y + beta*X)*(X**2 + Y**2)
    return X_dot, Y_dot

def Hopf_RK2(mu, w0, alpha, beta, X, Y, dt, D):
    X_noise = ((2*D*dt)**0.5)*random.gauss(0.0, 1.0)
    Y_noise = ((2*D*dt)**0.5)*random.gauss(0.0, 1.0)
    Xk1, Yk1 = Hopf_dots(mu, w0, alpha, beta, X                   , Y                   )
    Xk2, Yk2 = Hopf_dots(mu, w0, alpha, beta, X + Xk1*dt + X_noise, Y + Yk1*dt + Y_noise)
    new_X = X + (dt/2)*(Xk1 + Xk2) + X_noise
    new_Y = Y + (dt/2)*(Yk1 + Yk2) + Y_noise
    return new_X, new_Y

def HOPF(N, dt, mu, w0, alpha, beta, D, r0, phi0):
    X = np.zeros(N, dtype=float)
    Y = np.zeros(N, dtype=float)
    X[0] = r0*np.cos(phi0)
    Y[0] = r0*np.sin(phi0)
    for i in range(1, N):
        X[i], Y[i] = Hopf_RK2(mu, w0, alpha, beta, X[i-1], Y[i-1], dt, D)
    return X, Y


D = 0.001  #Noise strength
N = 30000
dt = 0.001
mu = 2
w0 = 1
alpha = 1
beta = 1

x, y = HOPF(N, dt, mu, w0, alpha, beta, D, 0, 0)
x2, y2 = HOPF(N, dt, mu, w0, alpha, beta, D, 0.1, 0)
x3, y3 = HOPF(N, dt, mu, w0, alpha, beta, D, 0.1, np.pi/2)
t = np.linspace(0, 100, 1000)
x4 = (np.sqrt(mu)) * np.cos(t)
y4 = (np.sqrt(mu)) * np.sin(t)
plt.figure()
plt.plot(x, y)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.plot(x4, y4)
plt.xlabel('time step')
plt.ylabel('x(t)')

plt.show()
