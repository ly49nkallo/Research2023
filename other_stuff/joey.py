import numpy
import matplotlib.pyplot

C_min = 1
C_max = 1 # maybe?
tau_T_null = 0
tau_gs = 1
tau_a = 1 # maybe?
Cm = 1
x_c = 0
X_hb = 1
S_min = 0
C_gs = 1000
tau_hb = 1
S_max = 0.5

def model(
        X_hb, # hair bundle position
        X_a, # adaption motor position
        p_m, 
        p_gs, 
        p_t,
        
        U_gsmax,
        X_a,
        k_gsmin,
        tau_m,
        delta_E_null
    ):

    k_gs = 1 - (p_s * (1 - k_gsmin))
    x_gs = (X_hb * x_gs) - (X_a * x_a) + x_c
    F_gs = k_gs * (x_gs - p_t)
    C = 1 - (p_m * (1 - C_min))
    S = S_min - (p_m * (1 - S_min))

    x_hb_dot = - (F_gs + x_hb) / tau_hb
    x_a_dot = (S_max * S * (F_gs - x_a) - (C_max * C)) / tau_a
    pm_dot = (C_m







