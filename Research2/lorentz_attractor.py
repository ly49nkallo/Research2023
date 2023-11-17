import numpy as np
import matplotlib.pyplot as plt

def dxdt(sigma, x, y):
    return sigma * (y - x)

def dydt(rho, x, y, z):
    return x * (rho - z) - y

def dzdt(beta, x, y, z):
    return x * y - beta * z

def get_default_lorentz_parameters():
    return {'sigma': 10,
            'beta' : 8/3,
            'rho'  : 28}
    
if __name__ == '__main__':
    sigma, beta, rho = get_default_lorentz_parameters().values()
    dt = 0.01
    time_steps = 10
    hist = []
    for t in range(0, time_steps, dt):
        pass