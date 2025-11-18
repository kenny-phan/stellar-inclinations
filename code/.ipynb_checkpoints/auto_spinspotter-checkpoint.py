import os
import sys
import numpy as np
import pandas as pd
from spinspotter import *

# Get SLURM array index
task_id = int(sys.argv[1])

ROOT = os.path.abspath("..") 
SRC_DIR = os.path.join(ROOT, "code")
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(ROOT, "data")

# Load input list
hwo_list = pd.read_csv(
    os.path.join(DATA_DIR, "hwo_exp.csv"),
    sep=',', skiprows=60, header=1
)

names = hwo_list["hip_name"]

# Pick the star based on array index
starname_raw = names.iloc[task_id]
starname = starname_raw.replace(" ", "-")

# Output folder for just this star
out_folder = os.path.join(DATA_DIR, "spinspotter_data")
os.makedirs(out_folder, exist_ok=True)

_, tot_process_result = ss_tutorial(starname_raw, plot=False, verbose=False)
    
if tot_process_result is None:
    print(f"No lightcurves found for {starname_raw}, skipping.")
else:
    good_fit = ss_check_fit(tot_process_result)
    np.savez(os.path.join(out_folder, f"{starname}.npz"), good_fit)

np.savez(os.path.join(out_folder, f"{starname}.npz"), good_fit)



