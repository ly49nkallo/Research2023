import matplotlib.pyplot as plt
r = 1
K = 100
x0 = 1

# x_t = 2x_{t-1} - (x_{t-1})^2
def x(x_prev, y_prev):
    return -2 * x_prev + x_prev ** 2

# y_t = y_{t-1} * (x_{t-1} - 1)
def y(x_prev, y_prev):
    return y_prev * (x_prev - 1)

stable_x_y = []

for i in range(-1000, 1000):
    for j in range(-1000, 1000):
        x_prev = i
        y_prev = j
        x_curr = x(x_prev, y_prev)
        y_curr = y(x_prev, y_prev)
        if (x_curr == x_prev and y_curr == y_prev):
            stable_x_y.append((x_curr, y_curr))
        
print(stable_x_y)