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

# support stuff
import time
import os
import importlib

def try_block(hd_name, plot=False, verbose=False):
    print(f"Beginning period analysis of {hd_name}")
    # Option 1: Search and download data using LightKurve.
    search_result = lk.search_lightcurve(hd_name)
    # print(search_result)
    
    # Download
    collection = search_result.download_all()

    p_avg_arr, p_err_arr = [], []

    for i in range(len(collection)):
        if verbose:
            print(f"using collection {i+1} of {len(collection)}")
        lc = collection[i]
    
        # process_LightCurve is where the bulk of the processing takes place
        fits_result, process_result = ss.process_LightCurve(lc, bs=120)

        if plot:
            # Plot, to make sure it looks correct
            collection.stitch().plot()
            plt.show()
            # plot our results, setting the show_peaks keyword to True to check how our parabola fits looks
            ss.plot_acf(fits_result, process_result, plot_peaks=True)
            plt.show()
    
        # print a summary of our statistics
        # ss.print_summary(fits_result, process_result, bs=120)
        p_avg = ss.bins_to_days(process_result['P_avg'], bs=120)
        p_err = ss.bins_to_days(process_result['P_err'], bs=120)

        p_avg_arr.append(p_avg)
        p_err_arr.append(p_err)
    
    return hd_name, p_avg_arr, p_err_arr

def ss_tutorial(hd_name, plot=False, verbose=False):
    
    try:
        hd_name, p_avg_arr, p_err_arr = try_block(hd_name, plot=plot, verbose=verbose)
    
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            print(f"{hd_name}: 'NoneType' object is not subscriptable. Trying {hd_name[:-2]}")
            hd_name, p_avg_arr, p_err_arr = try_block(hd_name[:-2], collection, plot=plot)
        else:
            raise
    return np.array([p_avg_arr, p_err_arr])