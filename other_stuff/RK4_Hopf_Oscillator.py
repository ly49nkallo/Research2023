import numpy as np
import matplotlib.pyplot as plt

def Hopf_dots(mu, w0, X, Y):
    X_dot = mu*X - w0*Y - X*(X**2 + Y**2)
    Y_dot = mu*Y + w0*X - Y*(X**2 + Y**2)
    return X_dot, Y_dot

def Hopf_RK4(mu, w0, X, Y, dt):
    Xk1, Yk1 = Hopf_dots(mu, w0, X           , Y           )
    Xk2, Yk2 = Hopf_dots(mu, w0, X + Xk1*dt/2, Y + Yk1*dt/2)
    Xk3, Yk3 = Hopf_dots(mu, w0, X + Xk2*dt/2, Y + Yk2*dt/2)
    Xk4, Yk4 = Hopf_dots(mu, w0, X + Xk3*dt  , Y + Yk3*dt  )
    new_X = X + (dt/6)*(Xk1 + 2*Xk2 + 2*Xk3 + Xk4)
    new_Y = Y + (dt/6)*(Yk1 + 2*Yk2 + 2*Yk3 + Yk4)
    return new_X, new_Y

dt = 0.001  #time step
N = 10000  #number of time steps
x = np.zeros(N, dtype=float)
y = np.zeros(N, dtype=float)

mu = 1  #setting parameters
w0 = 5

x[0], y[0] = 0.1, 0   #setting initial conditions

#calculating all the other points
for i in range(1, N):
    x[i], y[i] = Hopf_RK4(mu, w0, x[i-1], y[i-1], dt)

#plotting results
plt.figure()
plt.plot(x, y)
plt.xlabel('x(t)')
plt.ylabel('y(t)')

plt.figure()
plt.plot(x)
plt.xlabel('time step')
plt.ylabel('x(t)')
plt.show()

# plt.figure()

