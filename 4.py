# https://math.libretexts.org/Bookshelves/Scientific_Computing_Simulations_and_Modeling/Book%3A_Introduction_to_the_Modeling_and_Analysis_of_Complex_Systems_(Sayama)/09%3A_Chaos/9.03%3A_Lyapunov_Exponent
import numpy as np
import matplotlib.pyplot as plt

def log_dFdx(x):
    return np.log(np.abs(1 - 2*x))

def f(x, r):
    return x + r - (x ** 2)

def lyapunov_exponent(r):
    x = 0.1
    result = [log_dFdx(x)]
    for t in range(100):
        x = f(x, r)
        result.append(log_dFdx(x))
    return np.mean(result)

r_values = np.arange(0, 2, 0.01)
lambdas = [lyapunov_exponent(r) for r in r_values]
plt.plot(r_values, lambdas)
plt.xlabel('r')
plt.ylabel('Lyapunov Exponent')
plt.plot([0, 2], [0, 0])
plt.show()