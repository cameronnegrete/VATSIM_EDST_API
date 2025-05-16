import pygrib
import requests
import numpy as np
import json
from datetime import datetime
import os
import csv

# Function to download the latest RAP GRIB2 file based on the current date and cycle hour
def download_rap_grib2(date_str, cycle_hour, forecast_hour, save_path):
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

# Function to interpolate U, V wind components and temperature to flight levels and save to CSV files
def interpolate_uv_temp_at_flight_levels(grib_file):
    grbs = pygrib.open(grib_file)

    # Select required GRIB records
    hgt_grbs = grbs.select(name='Geopotential Height', typeOfLevel='isobaricInhPa')
    u_grbs = grbs.select(name='U component of wind', typeOfLevel='isobaricInhPa')
    v_grbs = grbs.select(name='V component of wind', typeOfLevel='isobaricInhPa')
    t_grbs = grbs.select(name='Temperature', typeOfLevel='isobaricInhPa')

    # Convert selected GRIB records to 3D NumPy arrays
    hgt_3d = np.array([g.values for g in hgt_grbs])
    u_3d = np.array([g.values for g in u_grbs])
    v_3d = np.array([g.values for g in v_grbs])
    t_3d = np.array([g.values for g in t_grbs]) - 273.15  # Convert Kelvin to Celsius

    # Get lat/lon arrays from GRIB
    lats, lons = hgt_grbs[0].latlons()

    # Convert geopotential height from meters to feet and then to flight levels
    hgt_ft = hgt_3d * 3.281
    fl_3d = hgt_ft / 100.0

    # Create output directory for CSV files
    output_folder = 'flight_level_csvs'
    os.makedirs(output_folder, exist_ok=True)

    # Target flight levels (FL000 to FL500 in steps of 10)
    fl_target = np.arange(0, 510, 10)

    # Loop through each target flight level and interpolate
    for fl in fl_target:
        temp_2d = np.full(lats.shape, np.nan)
        speed_2d = np.full(lats.shape, np.nan)
        dir_2d = np.full(lats.shape, np.nan)

        # Interpolate for each grid point
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                profile_fl = fl_3d[:, i, j]
                profile_u = u_3d[:, i, j]
                profile_v = v_3d[:, i, j]
                profile_t = t_3d[:, i, j]

                # Skip if any NaNs in the vertical profile
                if (np.any(np.isnan(profile_fl)) or np.any(np.isnan(profile_u)) or
                    np.any(np.isnan(profile_v)) or np.any(np.isnan(profile_t))):
                    continue

                # Interpolate values to the desired flight level
                u_val = np.interp(fl, profile_fl[::-1], profile_u[::-1])
                v_val = np.interp(fl, profile_fl[::-1], profile_v[::-1])
                t_val = np.interp(fl, profile_fl[::-1], profile_t[::-1])

                # Calculate wind speed and direction
                speed = np.sqrt(u_val**2 + v_val**2)
                direction = (270 - np.degrees(np.arctan2(v_val, u_val))) % 360

                # Store rounded integer values
                temp_2d[i, j] = int(round(t_val))
                speed_2d[i, j] = int(round(speed))
                dir_2d[i, j] = int(round(direction))

        # Define file paths for CSV outputs
        temp_csv_file_path = os.path.join(output_folder, f"FL{int(fl):03}-Temp.csv")
        dir_csv_file_path = os.path.join(output_folder, f"FL{int(fl):03}-Dir.csv")
        spd_csv_file_path = os.path.join(output_folder, f"FL{int(fl):03}-Spd.csv")

        # Write 2D temperature array to CSV
        with open(temp_csv_file_path, 'w', newline='') as temp_csvfile:
            writer = csv.writer(temp_csvfile)
            for row in temp_2d:
                writer.writerow([int(round(val)) if not np.isnan(val) else '' for val in row])

        # Write 2D wind direction array to CSV
        with open(dir_csv_file_path, 'w', newline='') as dir_csvfile:
            writer = csv.writer(dir_csvfile)
            for row in dir_2d:
                writer.writerow([int(round(val)) if not np.isnan(val) else '' for val in row])

        # Write 2D wind speed array to CSV
        with open(spd_csv_file_path, 'w', newline='') as spd_csvfile:
            writer = csv.writer(spd_csvfile)
            for row in speed_2d:
                writer.writerow([int(round(val)) if not np.isnan(val) else '' for val in row])

    grbs.close()

# Function to get the current RAP forecast date and time, and build file path info
def get_date():
    now = datetime.utcnow()
    date_str = now.strftime('%Y%m%d')
    cycle_hour = '12' if now.hour >= 12 else '00'
    forecast_hour = '00'
    save_path = 'rap_latest.grib2'

    # Store state in JSON format
    output = {'String': date_str, 'Cycle': cycle_hour, 'Forecast Hour': forecast_hour}
    json.dumps('wx_state.json', output, indent=4)

    return date_str, cycle_hour, forecast_hour, save_path

# Function to check if new forecast hour differs from previously stored state
def check_state(new_forecast):
    f = open('state.json')
    data = json.load(f)

    if new_forecast != data['Forecast Hour']:
        output = True
    else:
        output = False
    return output

# Wrapper function to execute the full workflow conditionally based on forecast state
def run_grid():
    date_str, cycle_hour, forecast_hour, save_path = get_date()
    state_check = check_state(forecast_hour)

    if state_check == True:
        download_rap_grib2(date_str, cycle_hour, forecast_hour, save_path)
        interpolate_uv_temp_at_flight_levels(save_path)
        output = 'Completed'
    else:
        output = 'Error, forecast is the same as previous forecast'

    return output
