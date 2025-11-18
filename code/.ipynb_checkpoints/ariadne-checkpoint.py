import astropy.units as u

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

def get_gaia_id_from_name(name):
    tab = Simbad.query_objectids(name)
    entry = [id for id in tab['id'] if id.startswith('Gaia DR3')][0]
    return int(entry[9:])

def get_gaia_id_by_coord(ra, dec):
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    radius = 5*u.arcsec
    job = Gaia.cone_search_async(coord, radius=radius)
    results = job.get_results()
    
    if len(results) == 0:
        return None
    
    # compute angular separations
    gaia_coords = SkyCoord(ra=results['ra']*u.deg, dec=results['dec']*u.deg)
    sep = coord.separation(gaia_coords)
    
    # pick the nearest source
    nearest_idx = sep.argmin()
    return results['source_id'][nearest_idx]