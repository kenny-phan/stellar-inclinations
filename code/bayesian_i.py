## From Equation 11 of Masuda and Winn 2020
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm

import matplotlib.pyplot as plt

from style import *

eps = 1e-3

def get_velocity(R, P):
    return 2*np.pi*R/P

def get_sig_velocity(R, P, R_err, P_err):
    return 2*np.pi*np.sqrt((R_err/P)**2 + (R*R_err/P**2)**2)

def L_x_asym(x, mu, sigma_l, sigma_r):
    return np.where(
        x < mu,
        norm(mu, sigma_l).pdf(x) * 2*sigma_l/(sigma_l+sigma_r),
        norm(mu, sigma_r).pdf(x) * 2*sigma_r/(sigma_l+sigma_r),
    )

# Uniform prior on v between 0 and 20
def P_v(v):
    return (v >= 0) & (v <= 20)

# Uniform prior on cos(i)
def P_cosi(cosi):
    return 1.0 if (0 <= cosi <= 1) else 0.0

# --- Integrand over u ---
def integrand(u, cosi, v_obs, u_obs, v_sigma_l, v_sigma_u, u_sigma_l, u_sigma_u):
    s = np.sqrt(1 - cosi**2)
    v = u / s

    # Handle zero uncertainty for v
    if v_sigma_l == 0 and v_sigma_u == 0:
        sigma_v = eps  # tiny width
        L_v = np.exp(-0.5 * ((v - v_obs)/sigma_v)**2) / (sigma_v * np.sqrt(2*np.pi))
    else:
        L_v = L_x_asym(v, v_obs, v_sigma_l, v_sigma_u)

    # Handle zero uncertainty for u
    if u_sigma_l == 0 and u_sigma_u == 0:
        sigma_u = eps  # tiny width
        L_u = np.exp(-0.5 * ((u - u_obs)/sigma_u)**2) / (sigma_u * np.sqrt(2*np.pi))
    else:
        L_u = L_x_asym(u, u_obs, u_sigma_l, u_sigma_u)

    if not np.isfinite(L_v) or not np.isfinite(L_u):
        print("BAD PDF:", u, cosi, v, L_v, L_u)

    return L_v * L_u * P_v(v)


# --- Posterior up to normalization ---
def p_cosi(cosi, v_obs, u_obs, v_sigma_l, v_sigma_u, u_sigma_l, u_sigma_u):
    if cosi <= 0 or cosi >= 1:
        return 0.0
    s = np.sqrt(1 - cosi**2)
    integral, _ = quad(integrand, 0, 20, 
                       args=(cosi, v_obs, u_obs, v_sigma_l, v_sigma_u, u_sigma_l, u_sigma_u))
    # print(integral)
    return P_cosi(cosi) * integral / s

def get_cosi(v_obs_arr, u_obs, v_sigma_l_arr, v_sigma_u_arr, u_sigma_l, u_sigma_u, 
             cosi_grid=np.linspace(0.01, 0.99, 200), plot=False, name=None):
    if plot:
        plt.figure(figsize=(6,4))
    posteriors = []
    for j in range(len(v_obs_arr)):
        v_obs, v_sigma_l, v_sigma_u = v_obs_arr[j], v_sigma_l_arr[j], v_sigma_u_arr[j]
        posterior = np.array([p_cosi(c, v_obs, u_obs, v_sigma_l, v_sigma_u, u_sigma_l, u_sigma_u) for c in tqdm(cosi_grid)])
        posterior /= np.trapezoid(posterior, cosi_grid)  # normalize
        posteriors.append(posterior)
    
    # --- Plot ---
        if plot:
            label = fr'$v = {v_obs:.2f}^{{+{np.abs(v_sigma_u - v_obs):.2f}}}_{{-{np.abs(v_obs - v_sigma_l):.2f}}},  v sin(i) = {u_obs:.2f}^{{+{np.abs(u_sigma_u - u_obs):.2f}}}_{{-{np.abs(u_obs - u_sigma_l):.2f}}}$'
            plt.plot(cosi_grid, posterior, lw=2, label=label, color=plot_colors_rgb[j])
    if plot:
        plt.xlabel(r'$\cos i$')
        plt.ylabel(r'$p(\cos i \mid D)$')
        plt.title(f'cos(i) for {name if name is not None else "this star"}')
        plt.legend()
        plt.show()

    return np.array(posteriors)

## error bounds on cos i (many ways)
def hpd(x, pdf, mass=0.68):
    """
    Compute the shortest (minimum width) interval [a, b] containing
    the given probability mass (e.g., 0.68) for an arbitrary tabulated PDF.
    
    Parameters
    ----------
    x : array of shape (N,)
        Grid values (must be increasing).
    pdf : array of shape (N,)
        Probability density (does not need to be normalized).
    mass : float
        Fraction of total probability to include (default=0.68).
    
    Returns
    -------
    a, b : floats
        Lower and upper bounds of the HPD interval.
    median : float
        Median of the distribution.
    a_err, b_err : floats
        Errors relative to the median: a_err = a - median, b_err = b - median.
    """

    # Normalize the PDF
    pdf = np.maximum(pdf, 0)  # avoid negatives
    pdf /= np.trapz(pdf, x)

    # Compute CDF on the same grid
    cdf = np.cumsum(pdf * np.gradient(x))
    cdf /= cdf[-1]

    N = len(x)
    best_width = np.inf
    best_i = None
    best_j = None

    j = 0
    for i in range(N):
        # move j forward until accumulated mass >= desired mass
        while j < N and (cdf[j] - cdf[i]) < mass:
            j += 1
        if j >= N:
            break
        width = x[j] - x[i]
        if width < best_width:
            best_width = width
            best_i, best_j = i, j

    # Extract interval
    # a = x[best_i]
    # b = x[best_j]

    # # Compute median
    median = np.interp(0.5, cdf, x)
    # low_err, max_err = np.abs(a - median), np.abs(b -median)
    upr, lwr = x[best_j], x[best_i]
    return median, lwr, upr 

def get_i(posterior, cosi_grid=np.linspace(0.01, 0.99, 200), plot=False, name=None):
    posterior = np.asarray(posterior).flatten()
    cosi_vals = hpd(cosi_grid, posterior) # in cosi
    # print(cosi_vals)
    i_vals = np.arccos(cosi_vals)
    i_vals_degrees = np.degrees(i_vals)
    _, i_upr, i_lwr = i_vals_degrees[0], i_vals_degrees[1], i_vals_degrees[2]
    i_med = np.degrees(np.arccos(cosi_grid[np.argmax(posterior)])) #reversed bc we flip the axis in cosi --> i
    # print(i_med, i_lwr, i_upr)
    if plot:
        x_i = np.degrees(np.arccos(cosi_grid))

        plt.plot(x_i, -posterior / np.trapz(posterior, x_i))   # normalized

        plt.axvline(i_lwr, color='C2', linestyle='-',  label='HDI 1σ')
        plt.axvline(i_upr, color='C2', linestyle='-')
        plt.axvline(i_med, color='k', linestyle=':',  label='Maximum')
        plt.legend()
        plt.xlabel(r'$i [^\circ]$')
        plt.ylabel('Normalized Probability Density')
        plt.title(fr'{name if name is not None else ''} $i={i_med:.2f}^{{+{np.abs(i_upr - i_med):.2f}}}_{{-{np.abs(i_med - i_lwr):.2f}}}$')
        plt.xlim(0, 90)
        
        plt.show()
    return i_med, i_lwr, i_upr