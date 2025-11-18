import os
import sys
import pandas as pd
import astropy.units as u

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter

from tap_query import *
from ariadne import *

# Get SLURM array index
task_id = int(sys.argv[1])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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

ra = hwo_list["ra"].iloc[task_id]
dec = hwo_list["dec"].iloc[task_id]

# Output folder for just this star
out_folder = os.path.join(DATA_DIR, "ariadne_data", starname)
os.makedirs(out_folder, exist_ok=True)

# Get Gaia ID
gaia_id = get_gaia_id_from_name(starname_raw)
if gaia_id is None:
    gaia_id = get_gaia_id_by_coord(ra, dec)

# Setup ARIADNE fitter
engine = 'dynesty'
nlive = 500
dlogz = 0.5
bound = 'multi'
sample = 'rwalk'
threads = 4
dynamic = False

setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]
models = ['phoenix','btsettl','btnextgen','btcond','kurucz','ck04']

f = Fitter()
f.setup = setup
f.av_law = 'fitzpatrick'
f.bma = True
f.models = models
f.n_samples = 1000

f.prior_setup = {
    'teff': ('default'),
    'logg': ('default'),
    'z': ('default'),
    'dist': ('default'),
    'rad': ('default'),
    'Av': ('default')
}

# Create star object
s = Star(starname, ra, dec, g_id=gaia_id, ignore=['2MASS'])

f.star = s
f.out_folder = out_folder

# Run fitting
f.initialize()
f.fit_bma()

print(f"Finished star {task_id}: {starname}")
