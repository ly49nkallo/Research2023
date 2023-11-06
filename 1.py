import matplotlib.pyplot as plt
import numpy as np

x0 = 1
y0 = 1

def x(x_prev, y_prev):
    return 0.5 * x_prev + y_prev

def y(x_prev, y_prev):
    return -0.6 * x_prev + y_prev

x_hist = []
y_hist = []

x_curr, y_curr = x0, y0

x_hist.append(x_curr)
y_hist.append(y_curr)

ts = 30

for t in range(1, ts):
    x_prev = x_curr
    y_prev = y_curr
    x_curr = x(x_prev, y_prev)
    y_curr = y(x_prev, y_prev)
    x_hist.append(x_curr)
    y_hist.append(y_curr)

plt.figure()
plt.plot(x_hist, 'g--')
plt.plot(y_hist)

#plot phase space

plt.figure()
plt.plot(x_hist, y_hist)
plt.show()