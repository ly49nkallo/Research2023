import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

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
    
# demonstation of noisy Hopf Oscillator
def show_demonstration(
        D = 0.000,
        N = 30000,
        dt = 0.001,
        mu = 2,
        w0 = 1,
        alpha = 1,
        beta = 1
    ):

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
    
def exp(t, a, lyp):
    return a * np.exp(lyp * t) - a + 1e-2
    
def lyapunov_exponent(N=30000, dt=0.01, mu=2, w0=1, alpha=1, beta=1, D=0, r0=0.1, phi0=0, epsilon=1e-2, plot=False):
    X1, Y1 = HOPF(N, dt, mu, w0, alpha, beta, D, r0, phi0)
    X2, Y2 = HOPF(N, dt, mu, w0, alpha, beta, D, r0 + epsilon, phi0)
    d = np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)[:5000]
    try:
        params, cv = curve_fit(exp, np.arange(0, 5000), d, p0 = (1, -1))
    except RuntimeError:
        print('Runtime Error passed')
        return None
    if not np.isfinite(np.sum(cv)):
        print('false')
        try:
            params, cv = curve_fit(exp, np.arange(0, 5000), d, p0 = (1, 1))
        except RuntimeError as e:
            print('Runtime Error passed')
            return None
    if plot:
        Y3 = np.array(exp(np.arange(0, 5000), *params))
        print(params[1], cv)
        plt.figure()
        plt.plot(d)
        plt.plot(Y3)
        plt.xlabel("time")
        plt.ylabel("distance")
        plt.title("Distance of trajectories")
        plt.show()
    return params[1]

def plot_lyapunov_exponents(N):
    mus = np.linspace(0.255, 0.28, 100)
    lambdas = []
    for mu in tqdm(mus):
        lambdas.append(lyapunov_exponent(N=N, mu = mu, r0 = 1 / 2, plot=mu == mus[43] or mu == mus[0] or mu == mus[-1]))
    print(list(zip(mus, lambdas)))
    plt.figure()
    plt.plot(mus, lambdas)
    plt.show()
        
def main():
    N = 30000
    # show_demonstration(N=N, dt=0.01, mu=0.265, w0=1, alpha=1, beta=1, D=0)
    plot_lyapunov_exponents(N)
    plt.show()
if __name__ == '__main__':
    main()