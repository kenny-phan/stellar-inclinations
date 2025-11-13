## From Equation 11 of Masuda and Winn 2020
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

from style import *

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

def plot_cosi(df, v_obs_arr, u_obs, v_sigma_arr, u_sigma, idx_unique, k=0, cosi_grid=np.linspace(0.01, 0.99, 200)):
    
    plt.figure(figsize=(6,4))
    
    for j in range(len(v_obs_arr)):
        v_obs, v_sigma = v_obs_arr[j], v_sigma_arr[j]
        posterior = np.array([p_cosi(c, v_obs_arr[j], u_obs, v_sigma_arr[j], u_sigma) for c in cosi_grid])
        posterior /= np.trapezoid(posterior, cosi_grid)  # normalize
    
        # print(f"most probable cos i for index {j} = {cosi_grid[np.argmax(posterior)]}")
        # print(f"most probable i = {np.degrees(np.arccos(cosi_grid[np.argmax(posterior)]))} degrees")
    
    # --- Plot ---
        plt.plot(cosi_grid, posterior, lw=2, label=fr'$v = {v_obs:.2f} \pm {v_sigma:.2f}, v sin(i) = {u_obs} \pm {u_sigma}$', color=plot_colors_rgb[j])
    
    plt.xlabel(r'$\cos i$')
    plt.ylabel(r'$p(\cos i \mid D)$ (normalized)')
    plt.title(f'cos(i) for {df['hd_name'].iloc[idx_unique[k]]}')
    plt.legend()
    plt.show()