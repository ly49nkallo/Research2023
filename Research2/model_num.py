import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from tqdm import tqdm
import os

def kgs_func(pgs: float, kgsmin: float) -> float:
    return 1 - pgs * (1 - kgsmin)


def xgs_func(xhb: float, xa: float, chihb: float, chia: float, xc: float) -> float:
    return chihb * xhb - chia * xa + xc


def Fgs_func(pT: float, kgs: float, xgs: float) -> float:
    return kgs * (xgs - pT)


def C_func(pm: float, Cmin: float) -> float:
    return 1 - pm * (1 - Cmin)


def S_func(pm: float, Smin: float) -> float:
    return Smin + pm * (1 - Smin)


def pTinf_func(kgs: float, xgs: float, Ugsmax: float, dE0: float) -> float:
    return 1 / (1 + np.exp(Ugsmax * (dE0 - kgs * (xgs - 1 / 2))))


def derivative(x: list, p: list) -> np.ndarray:
    '''
    Calculute the derivatve of parameters at the current timestep W.R.T. time.
    @Params
        x:list
            a list of the current timestep's values for the five parameters
        p:list
            a list of the hyperparameters (unchanging)
    @Returns
        list
            the derivative of the current timestep's W.R.T. time.
    '''
    xhb, xa, pm, pgs, pT = x
    (
        tauhb0,
        taum0,
        taugs0,
        tauT0,
        Cmin,
        Smax,
        Smin,
        Cm,
        Cgs,
        Ugsmax,
        dE0,
        kgsmin,
        chihb,
        chia,
        xc,
    ) = p

    Cmax = 1 - Smax

    kgs = kgs_func(pgs, kgsmin)
    xgs = xgs_func(xhb, xa, chihb, chia, xc)
    Fgs = Fgs_func(pT, kgs, xgs)
    C = C_func(pm, Cmin)
    S = S_func(pm, Smin)

    pT_inf = pTinf_func(kgs, xgs, Ugsmax, dE0)

    dxhb = -(Fgs + xhb) / tauhb0
    dxa = Smax * S * (Fgs - xa) - Cmax * C
    dpm = (Cm * pT * (1 - pm) - pm) / taum0
    dpgs = (Cgs * pT * (1 - pgs) - pgs) / taugs0
    dpT = (pT_inf - pT) / tauT0

    return [dxhb, dxa, dpm, dpgs, dpT]

def calculate_data(x0:list, timesteps:int, dt:float, param_file:str='./model_values.json', method:str='RK2') -> np.ndarray:
    '''
    Calculate, record, and return all of the parameter values in a full simulation.
    @Params
        x0:list
            the initial values of the variables
        timesteps:int
            the exact number of timesteps the simulation should run
        dt:float
            the integrating delta w.r.t. time.
    @Keyword
        param_file:str
            the directory path to a json file containing the model hyperparameters
        method:str
            How to integrate the trajectory; [Euler, RK2, RK3, RK4]
    @Returns
        np.ndarray
            Numpy array with shape (5, t), where t is the number of timesteps that are elapsed
    '''
    filepath = Path(param_file)
    with open(os.path.dirname(__file__) + os.sep  + './model_values.json', 'r') as f:
        p = json.load(f)
        params = list(p.values())
        param_names = tuple(p.keys())
    hist = None
    if method == 'RK2':
        hist = _RK2_calculate_data(x0, timesteps, dt, params)
    return hist

def _RK2_calculate_data(x0:list, timesteps:int, dt:float, params)->np.ndarray:
    timesteps = int(timesteps)
    logging.info("Beginning RK2 simulation")
    x = x0
    hist = np.zeros((timesteps, len(x0)), dtype=np.float32)
    '''
    for i in tqdm(range(1, n+1)):
        hist.append([x, y, z])
        xk1 = dxdt(x, y)
        yk1 = dydt(x, y, z)
        zk1 = dzdt(x, y, z)
        xk2 = dxdt(x + xk1 * dt, y + yk1 * dt)
        yk2 = dydt(x + xk1 * dt, y + yk1 * dt, z + zk1 * dt)
        zk2 = dzdt(x + xk1 * dt, y + yk1 * dt, z + zk1 * dt)
        x += (xk1 + xk2) / 2 * dt
        y += (yk1 + yk2) / 2 * dt
        z += (zk1 + zk2) / 2 * dt
    '''
    for i in tqdm(range(timesteps)):
        hist[i] = x
        k1 = derivative(x, params)
        k2 = derivative([a + b for a,b in zip(x, [d * dt for d in k1])], params)
        x =  [c + (a + b) / 2 * dt for a, b, c in zip(k1, k2, x)]

    return hist

if __name__ == '__main__':
    hist = []
    x0 = -1, 0, .5, .5, .5

    with open(os.path.dirname(__file__) + os.sep  + './model_values.json', 'r') as f:
        p = json.load(f)
        params = tuple(p.values())
        param_names = tuple(p.keys())
        params = np.array(params)
        # params[11] = 0.6

    x = list(x0)
    dt = 5e-1
    steps = 5e4

    hist = calculate_data(x, steps, dt)
    # params[8] = 900
    # steps_r = 1 / steps * 900

    # for i in tqdm(range(int(steps))):
    #     ret = derivative(x, params)
    #     # params[8] -= steps_r
    #     x = [x[i] + ret[i] * dt for i in range(len(x))]
    #     hist.append(tuple(x))
        
    # lyapunov exponent is the time average of log|dF/dx| over every state 
    th = int(100/dt)
    for i, name in enumerate(['xhb', 'xa', 'pm', 'pgs', 'pT']):
        plt.figure(i+1)
        plt.title(name)
        plt.plot([x[0] for x in hist[th:]], [x[i] for x in hist][th:])
        plt.savefig(f'./Phase_diagram {i}.png')
    sample_point = (hist[int(3*dt)][0], hist[int(3*dt)][1])
    # Find nearby points
    for x, y in zip([x[0] for x in hist[th:]], [x[i] for x in hist][th:]):
        pass
    print(sample_point)
    plt.show()