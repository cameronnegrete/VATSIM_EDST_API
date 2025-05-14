import pygrib
import requests
import numpy as np
import json
import time
from datetime import datetime

def download_rap_grib2(date_str, cycle_hour, forecast_hour, save_path):
    """
    Download a RAP GRIB2 file from NOMADS.

    - date_str: 'YYYYMMDD'
    - cycle_hour: '00', '03', ..., '21'
    - forecast_hour: '00', '01', ..., '18'
    """
    base_url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod'
    file_name = f"rap.t{cycle_hour}z.awp130pgrbf{forecast_hour.zfill(2)}.grib2"
    url = f"{base_url}/rap.{date_str}/{file_name}"

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download file. HTTP {response.status_code}: {url}")

def interpolate_uv_at_flight_levels(grib_file):
    """
    Interpolates U and V wind components to flight levels 0 to 450 (every FL10).
    Returns a dict of {flight_level: {'u': u_interp.tolist(), 'v': v_interp.tolist()}}.
    """
    start_time = time.time()

    grbs = pygrib.open(grib_file)

    # Get geopotential height for pressure levels
    hgt_grbs = grbs.select(name='Geopotential Height', typeOfLevel='isobaricInhPa')
    u_grbs = grbs.select(name='U component of wind', typeOfLevel='isobaricInhPa')
    v_grbs = grbs.select(name='V component of wind', typeOfLevel='isobaricInhPa')

    levels = sorted(set([g.level for g in hgt_grbs]), reverse=True)
    hgt_3d = np.array([g.values for g in hgt_grbs])
    u_3d = np.array([g.values for g in u_grbs])
    v_3d = np.array([g.values for g in v_grbs])

    # Assume all fields share the same lat/lon
    lats, lons = hgt_grbs[0].latlons()

    # Convert geopotential height (in gpm) to flight levels in hundreds of feet (approx)
    hgt_ft = hgt_3d * 3.281  # meters to feet (approx)
    fl_3d = hgt_ft / 100.0   # feet to flight level

    # Create storage for interpolated fields
    fl_target = np.arange(0, 510, 10)
    interpolated = {}

    for fl in fl_target:
        u_interp = np.empty_like(lats)
        v_interp = np.empty_like(lats)
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                profile_fl = fl_3d[:, i, j]
                profile_u = u_3d[:, i, j]
                profile_v = v_3d[:, i, j]
                if np.any(np.isnan(profile_fl)) or np.any(np.isnan(profile_u)) or np.any(np.isnan(profile_v)):
                    u_interp[i, j] = np.nan
                    v_interp[i, j] = np.nan
                else:
                    u_interp[i, j] = np.interp(fl, profile_fl[::-1], profile_u[::-1])
                    v_interp[i, j] = np.interp(fl, profile_fl[::-1], profile_v[::-1])
        interpolated[int(fl)] = {'u': u_interp.astype(int).tolist(), 'v': v_interp.astype(int).tolist()}

    grbs.close()

    end_time = time.time()
    print(f"Interpolation completed in {end_time - start_time:.2f} seconds")

    return interpolated

# Determine date and cycle based on current UTC time
now = datetime.utcnow()
date_str = now.strftime('%Y%m%d')
cycle_hour = '12' if now.hour >= 12 else '00'
forecast_hour = '00'
save_path = 'rap_latest.grib2'

download_rap_grib2(date_str, cycle_hour, forecast_hour, save_path)
interpolated_data = interpolate_uv_at_flight_levels(save_path)

# Extract lats/lons from one GRIB message (can reuse)
grbs = pygrib.open(save_path)
lats, lons = grbs.select(name='U component of wind', typeOfLevel='isobaricInhPa')[0].latlons()
grbs.close()

with open('uv.json', 'w') as fp:
    json.dump(interpolated_data, fp)
