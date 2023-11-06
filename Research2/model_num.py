import numpy as np
import matplotlib.pyplot as plt


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


def derivative(x: np.ndarray, p: np.ndarray) -> np.ndarray:
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

    return np.array([dxhb, dxa, dpm, dpgs, dpT])

if __name__ == '__main__':
    xhist = []
    x0 = -1, 0, .5, .5, .5
    import json
    from tqdm import tqdm
    import os

    with open(os.path.dirname(__file__) + os.sep  + './model_values.json', 'r') as f:
        p = json.load(f)
        params = tuple(p.values())
        params = np.array(params)
        params[11] = 0.6

    x = np.array(x0)
    dt = 1e-2
    steps = 2e5
    params[8] = 900

    steps_r = 1 / steps * 900

    for i in tqdm(range(int(steps))):
        ret = derivative(x, params)
        params[8] -= steps_r
        x += ret * dt
        xhist.append(x[0])
        
    # lyapunov exponent is the time average of log|dF/dx| over every state
    


    plt.figure()
    plt.plot(xhist[1000:])
    plt.show()