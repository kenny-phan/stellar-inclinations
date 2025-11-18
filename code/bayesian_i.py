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

def get_cosi(df, v_obs_arr, u_obs, v_sigma_arr, u_sigma, idx_unique, k=0, cosi_grid=np.linspace(0.01, 0.99, 200), plot=False):
    if plot:
        plt.figure(figsize=(6,4))
    posteriors = []
    for j in range(len(v_obs_arr)):
        v_obs, v_sigma = v_obs_arr[j], v_sigma_arr[j]
        posterior = np.array([p_cosi(c, v_obs_arr[j], u_obs, v_sigma_arr[j], u_sigma) for c in cosi_grid])
        posterior /= np.trapezoid(posterior, cosi_grid)  # normalize
        posteriors.append(posterior)
        # print(f"most probable cos i for index {j} = {cosi_grid[np.argmax(posterior)]}")
        # print(f"most probable i = {np.degrees(np.arccos(cosi_grid[np.argmax(posterior)]))} degrees")
    
    # --- Plot ---
        if plot:
            plt.plot(cosi_grid, posterior, lw=2, label=fr'$v = {v_obs:.2f} \pm {v_sigma:.2f}, v sin(i) = {u_obs} \pm {u_sigma}$', color=plot_colors_rgb[j])
    if plot:
        plt.xlabel(r'$\cos i$')
        plt.ylabel(r'$p(\cos i \mid D)$ (normalized)')
        plt.title(f'cos(i) for {df['hd_name'].iloc[idx_unique[k]]}')
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

    return median, x[best_j], x[best_i]  #median, lower bound, upper bound

def get_i(posterior, cosi_grid=np.linspace(0.01, 0.99, 200), plot=False):
    cosi_vals = hpd(cosi_grid, posterior) # in cosi
    # print(cosi_vals)
    i_vals = np.arccos(cosi_vals)
    i_vals_degrees = np.degrees(i_vals)
    i_med, i_lwr, i_upr = i_vals_degrees[0], i_vals_degrees[1], i_vals_degrees[2]
    # print(i_med, i_lwr, i_upr)
    if plot:
        x_i = np.degrees(np.arccos(cosi_grid))

        plt.plot(x_i, -posterior / np.trapz(posterior, x_i))   # normalized

        plt.axvline(i_lwr, color='C2', linestyle='-',  label='HPD 1σ')
        plt.axvline(i_upr, color='C2', linestyle='-')
        plt.axvline(i_med, color='k', linestyle=':',  label='median')
        plt.legend()
        plt.xlabel('i')
        plt.ylabel('pdf (normalized)')
        plt.title(fr'$i={i_med:.2f}^{{+{i_upr - i_med:.2f}}}_{{-{i_med - i_lwr:.2f}}}$')
        plt.show()
    return i_med, i_lwr, i_upr
