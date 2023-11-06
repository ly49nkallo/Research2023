import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 1) ** 3 + 1.2

result = []
x = 0.05
for i in range(100):
    result.append(x)
    x = f(x)

xmin, xmax = 0, 2
rng = np.arange(xmin, xmax, (xmax - xmin) / 100)
horizontal = [result[0]]
vertical = [result[0]]

for x in result[1:]:
    horizontal.append(vertical[-1])
    vertical.append(x)
    horizontal.append(x)
    vertical.append(x)
    
plt.plot(horizontal, vertical)
plt.plot(rng, list(map(f, rng)))
plt.plot([xmin, xmax], [xmin, xmax])
plt.show()