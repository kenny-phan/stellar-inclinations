## From Equation 11 of Masuda and Winn 2020
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

def get_velocity(R, P):
    return 2*np.pi*R/P

def get_sig_velocity(R, P, R_err, P_err):
    return 2*np.pi*np.sqrt((R_err/P)**2 + (R*R_err/P**2)**2)
    
# Gaussian likelihoods
# def L_v(v, v_obs, sigma_v):
#     return norm.pdf(v, loc=v_obs, scale=sigma_v)

# def L_u(u):
#     return norm.pdf(u, loc=u_obs, scale=sigma_u)

def L_x(x, x_obs, sigma):
    return norm.pdf(x, loc=x_obs, scale=sigma)

# Uniform prior on v between 0 and 20
def P_v(v):
    return (v >= 0) & (v <= 20)

# Uniform prior on cos(i)
def P_cosi(cosi):
    return 1.0 if (0 <= cosi <= 1) else 0.0

# --- Integrand over u ---
def integrand(u, cosi, v_obs, u_obs, v_sigma, u_sigma):
    s = np.sqrt(1 - cosi**2)
    v = u / s
    return L_x(v, v_obs, v_sigma) * L_x(u, u_obs, u_sigma) * P_v(v)

# --- Posterior up to normalization ---
def p_cosi(cosi, v_obs, u_obs, v_sigma, u_sigma):
    if cosi <= 0 or cosi >= 1:
        return 0.0
    s = np.sqrt(1 - cosi**2)
    integral, _ = quad(integrand, 0, 20, args=(cosi, v_obs, u_obs, v_sigma, u_sigma))
    return P_cosi(cosi) * integral / s