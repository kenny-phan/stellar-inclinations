# query exoplanet archive
import pandas as pd
import numpy as np
from io import StringIO
import requests
import json

# stolen from https://gist.github.com/pearsonkyle/4aee6976a867d643251ca97ee03edaee
def tap_query(base_url, query, dataframe=True):
    # table access protocol query

    # build url
    uri_full = base_url
    for k in query:
        if k != "format":
            uri_full+= "{} {} ".format(k, query[k])
    
    uri_full = uri_full[:-1] + "&format={}".format(query.get("format","csv"))
    uri_full = uri_full.replace(' ','+')
    print(uri_full)

    response = requests.get(uri_full, timeout=90)
    # TODO check status_code? 

    if dataframe:
        return pd.read_csv(StringIO(response.text))
    else:
        return response.text

pi = 3.14159
au=1.496e11 # m 
rsun = 6.955e8 # m
G = 0.00029591220828559104 # day, AU, Msun

# keplerian semi-major axis (au)
sa = lambda m,P : (G*m*P**2/(4*pi**2) )**(1./3) 

def new_scrape(query=None):

    uri_ipac_base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="

    if query is not None:
        uri_ipac_query = query
        default = tap_query(uri_ipac_base, uri_ipac_query)
    else:
        uri_ipac_query = {
            # Table columns: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
            "select"   : "pl_name,hd_name, hostname,tran_flag,pl_massj,pl_radj,pl_ratdor,"
                        "pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbeccen,"
                        "pl_orbincl,pl_orblper,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,"
                        "st_teff,st_met,st_logg,st_mass,st_rad,ra,dec",
            "from"     : "ps", # Table name
            "where"    : "tran_flag = 1 and default_flag = 1",
            "order by" : "pl_name",
            "format"   : "csv"
        }

        default = tap_query(uri_ipac_base, uri_ipac_query)
    
    # fill in missing columns
    uri_ipac_query['where'] = 'tran_flag=1'
    extra = tap_query(uri_ipac_base, uri_ipac_query)

    # for each planet
    for i in default.pl_name:

        # extract rows for each planet
        ddata = default.loc[default.pl_name==i]
        edata = extra.loc[extra.pl_name==i]

        # for each nan column in default
        nans = ddata.isna()
        for k in ddata.keys():
            if nans[k].bool(): # if col value is nan
                if not edata[k].isna().all(): # if replacement data exists
                    # replace with first index 
                    default.loc[default.pl_name==i,k] = edata[k][edata[k].notna()].values[0]
                    # TODO could use mean for some variables (not mid-transit)
                    # print(i,k,edata[k][edata[k].notna()].values[0])
                else:
                    # permanent nans - require manual entry
                    if k == 'pl_orblper': # omega
                        default.loc[default.pl_name==i,k] = 0
                    elif k == 'pl_ratdor': # a/R*
                        # Kepler's 3rd law
                        semi = sa(ddata.st_mass.values[0], ddata.pl_orbper.values[0])
                        default.loc[default.pl_name==i,k] = semi*au / (ddata.st_rad.values[0]*rsun)
                    elif k == 'pl_orbincl': # inclination
                        default.loc[default.pl_name==i,k] = 90
                    elif k == "pl_orbeccen": # eccentricity
                        default.loc[default.pl_name==i,k] = 0
                    elif k == "st_met": # [Fe/H]
                        default.loc[default.pl_name==i,k] = 0
    return default

def find_nan_indices(df, field_str, field_err_str1, field_err_str2, isolate=False): 
    """
    Identify indices in a DataFrame where the given field and its error field
    are NaN, regardless of NaNs in other fields.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns:
        'st_rad', 'st_raderr1', 'st_rotp', 'st_rotperr1', 'st_vsin', 'st_vsinerr1'
    field_str : str
        Name of main field to check for NaN
    field_err_str1 : str
        Name of error1 field to check for NaN
    field_err_str2 : str
        Name of error2 field to check for NaN
    isolate : boolean
        toggles whether you want the indices where the fields are *only* NaNs

    Returns
    -------
    this_nan_indices : list
        Indices where field_str AND field_err_str are NaN.
    other_nan_indices : list
        Indices where at least one NaN exists but not both target fields.
    """

    this_nan_indices = []
    other_nan_indices = []

    for i in range(len(df)):
        # extract all fields
        R      = df['st_rad'].iloc[i]
        R_err1  = df['st_raderr1'].iloc[i]
        R_err2  = df['st_raderr2'].iloc[i]
        
        P      = df['st_rotp'].iloc[i]
        P_err1  = df['st_rotperr1'].iloc[i]
        P_err2  = df['st_rotperr2'].iloc[i]

        u_obs  = df['st_vsin'].iloc[i]
        u_err1  = df['st_vsinerr1'].iloc[i]
        u_err2  = df['st_vsinerr2'].iloc[i]

        nan_fields = []
        if np.isnan(R):     nan_fields.append("st_rad")
        if np.isnan(R_err1): nan_fields.append("st_raderr1")
        if np.isnan(R_err2): nan_fields.append("st_raderr2")
            
        if np.isnan(P):     nan_fields.append("st_rotp")
        if np.isnan(P_err1): nan_fields.append("st_rotperr1")
        if np.isnan(P_err2): nan_fields.append("st_rotperr2")

        if np.isnan(u_obs): nan_fields.append("st_vsin")
        if np.isnan(u_err1): nan_fields.append("st_vsinerr1")
        if np.isnan(u_err2): nan_fields.append("st_vsinerr2")

        if isolate:
            if set(nan_fields) == {field_str, field_err_str1, field_err_str2}:
                this_nan_indices.append(i)
            elif nan_fields:
                other_nan_indices.append(i)
        else:
            if field_str in nan_fields and field_err_str1 in nan_fields and field_err_str2 in nan_fields:
                this_nan_indices.append(i)
            elif nan_fields:
                other_nan_indices.append(i)

    return this_nan_indices, other_nan_indices