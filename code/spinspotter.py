# Imports
# library import
import SpinSpotter as ss

# dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit
from tqdm import tqdm

# support stuff
import time
import os
import importlib

# stellar-inclination codes
from bayesian_i import *

# RUN SS

import lightkurve as lk
import matplotlib.pyplot as plt

import os
from lightkurve import LightkurveError

def safe_download(search_result):
    """
    Downloads all lightcurves from a SearchResult, skipping corrupted files.
    """
    collection = []
    for i in range(len(search_result)):
        try:
            lc = search_result[i].download()
            if lc is not None:
                collection.append(lc)
        except LightkurveError as e:
            # Try to find and remove the corrupted cached file
            msg = str(e)
            if "Not recognized as a supported data product" in msg:
                # Extract path from the error message
                start = msg.find("/home")
                end = msg.find(".fits") + 5
                if start != -1 and end != -1:
                    corrupt_path = msg[start:end]
                    if os.path.exists(corrupt_path):
                        print(f"Removing corrupted file: {corrupt_path}")
                        os.remove(corrupt_path)
            print(f"Skipping problematic file {i}: {e}")
    return collection

def try_block(hd_name, plot=False, verbose=False):
    print(f"Beginning period analysis of {hd_name}")
    
    try:
        search_result = lk.search_lightcurve(hd_name)
        exp_times = search_result.exptime.value.astype(int)
        if len(search_result) == 0:
            raise ValueError(f"No lightcurve found for {hd_name}")

        collection = safe_download(search_result) #.download_all()
        if collection is None or len(collection) == 0:
            raise ValueError(f"Download failed or returned empty collection for {hd_name}")

    except Exception as e:
        print(f"Error accessing lightcurve for {hd_name}: {e}")
        return hd_name, None, None 

    tot_fits_result, tot_process_result = [], []

    for i, lc in enumerate(collection):
        if verbose:
            print(f"Using collection {i+1} of {len(collection)}")
            print(f"Exposure time is {exp_times[i]}")
        
        try:
            # process_LightCurve is your main processing function
            fits_result, process_result = ss.process_LightCurve(lc, bs=exp_times[i])

            if plot:
                collection.stitch().plot()
                plt.show()
                ss.plot_acf(fits_result, process_result, plot_peaks=True)
                plt.show()

            tot_fits_result.append(fits_result)
            tot_process_result.append(process_result)

        except Exception as e:
            print(f"Error processing lightcurve {i+1} for {hd_name}: {e}")
            continue

    return hd_name, tot_fits_result, tot_process_result, exp_times


def ss_tutorial(hd_name, plot=False, verbose=False):
    # First attempt
    hd_name, tot_fits_result, tot_process_result, exp_times = try_block(
        hd_name, plot=plot, verbose=verbose
    )

    # If first attempt failed, try with hd_name[:-2]
    if (tot_fits_result is None or tot_process_result is None) and \
       isinstance(hd_name, str) and hd_name.endswith(("A", "B")):
        print(f"Could not process {hd_name}. Trying {hd_name[:-2]} instead.")
        hd_name, tot_fits_result, tot_process_result = try_block(
            hd_name[:-2], plot=plot, verbose=verbose
        )

    return tot_fits_result, tot_process_result, exp_times


def ss_check_fit(tot_process_result, exp_times, A_min=0.1, R_min=0.9):
    good_process_result, good_exptimes = [], []
    for i, process_result in enumerate(tot_process_result):
        if (process_result['A_avg'] > A_min) and (process_result['R_avg'] > R_min):
            good_process_result.append(process_result)
            good_exptimes.append(exp_times[i])
    return good_process_result, good_exptimes

    
## --- MAKE CUTS

def check_if_on_ms(Teff, logg):
    """
    Returns True if the star is on the main sequence, False otherwise.
    """
    if Teff >= 6000:
        return logg >= 3.5
    elif Teff <= 4250:
        return logg >= 4.0
    elif 4250 < Teff < 6000:
        return logg >= (5.2 - 2.8 * Teff * 1e-4)
    else:
        return False

def filter_ms_indices(df, idx_unique):
    """
    Given a DataFrame and a list/array of indices, returns only the indices
    where the star is on the main sequence.
    """
    keep = []
    for k in idx_unique:
        Teff = df['st_teff'].iloc[k]
        logg = df['st_logg'].iloc[k]
        if check_if_on_ms(Teff, logg):
            keep.append(k)
    return keep

# def filter_periods(test_data):
#     periods = [] 
#     for i in range(len(test_data[0])): 
#         if not np.isnan(test_data[0][i]): 
#             periods.append(test_data[:, i])
#     return np.array(periods)

# -- GET VALUES
def get_v_ss(df, periods, idx_unique, k):
    R, R_err = df['st_rad'].iloc[idx_unique[k]], df['st_raderr1'].iloc[idx_unique[k]]
        
    u_obs, u_sigma = df['st_vsin'].iloc[idx_unique[k]], df['st_vsinerr1'].iloc[idx_unique[k]]
    
    # P_arr, P_err_arr = [], []
    v_obs_arr, v_sigma_arr = [], []
    for ii, period in enumerate(periods):
        P, P_err = period[0], period[1]
        v_obs, v_sigma = get_velocity(R, P), get_sig_velocity(R, P, R_err, P_err)
        # P_arr.append(P)
        # P_err_arr.append(P_err)
        v_obs_arr.append(v_obs)
        v_sigma_arr.append(v_sigma)

    return np.array(v_obs_arr), np.array(v_sigma_arr), u_obs, u_sigma

def get_i_ss(df, hd_name, idx_unique, k, plot_ss=False, plot_i=False):
    
    _, tot_process_result, exp_times = ss_tutorial(hd_name, plot=plot_ss)

    periods = []
    for ll, this_process_result in enumerate(tot_process_result):
        p_avg = ss.bins_to_days(this_process_result['P_avg'], bs=exp_times[ll])
        p_err = ss.bins_to_days(this_process_result['P_err'], bs=exp_times[ll])
        periods.append([p_avg, p_err])
    
    # periods = filter_periods(test_data)

    i_med_arr, i_lwr_arr, i_upr_arr = [], [], []

    v_obs_arr, v_sigma_arr, u_obs, u_sigma = get_v_ss(df, periods, idx_unique, k)
    posteriors = get_cosi(df, v_obs_arr, u_obs, v_sigma_arr, u_sigma, idx_unique, k=k)

    for kk in tqdm(range(len(posteriors))):
                
        i_med, i_lwr, i_upr = get_i(posteriors[kk], plot=plot_i)
        i_med_arr.append(i_med)
        i_lwr_arr.append(i_lwr)
        i_upr_arr.append(i_upr)

    return np.array(i_med_arr), np.array(i_lwr_arr), np.array(i_upr_arr)

def get_best_spin_period(good_fit, good_exptimes):
    """
    Returns P_avg and P_err (in days) for the best entry in process_result
    according to:
      - largest A_avg
      - largest R_avg
      - smallest abs(0.5 - B_avg)
    
    Parameters
    ----------
    star_data : dict
        Loaded from consolidated spinspotter .npz, star_data['arr_0'][0] is a list of dicts
    bs : int
        bin size for SpinSpotter's bins_to_days conversion
    
    Returns
    -------
    P_avg_days, P_err_days : float
        Best period and error in days
    best_idx : int
        Index of the selected entry
    """
    process_result_list = good_fit#['arr_0']
    
    if len(process_result_list) == 0:
        return np.nan, np.nan, None
    
    A_avg_arr = np.array([res['A_avg'] for res in process_result_list])
    R_avg_arr = np.array([res['R_avg'] for res in process_result_list])
    B_avg_arr = np.array([res['B_avg'] for res in process_result_list])
    P_avg_arr = np.array([res['P_avg'] for res in process_result_list])
    
    B_score = np.abs(0.5 - B_avg_arr)
    
    # Compute a combined score: high A_avg, high R_avg, small abs(0.5 - B_avg)
    score = A_avg_arr + R_avg_arr - B_score

    mask = ~np.isnan(P_avg_arr)
    best_idx = np.argmax(score[mask])

    best_result = process_result_list[best_idx]
    best_exptime = good_exptimes[best_idx]
    
    P_avg_days = ss.bins_to_days(best_result['P_avg'], bs=best_exptime)
    P_err_days = ss.bins_to_days(best_result['P_err'], bs=best_exptime)
    
    return P_avg_days, P_err_days, best_idx
