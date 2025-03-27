import numpy as np
import pickle

# Load data
print("Loading the data...")
observation_array = np.load('/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/observation_array_jan_march.npy')  # Shape: (59, 60, 5695)

# Define parameters
num_days, num_lead_times, num_stations = observation_array.shape
num_ensembles = 51  # Number of ensembles in persistence forecast
######################
# Step 1: Create the Persistence Forecast (reading the fillup from a folder and taking the final timesteps)
#########################################
print("Reading fillup to generate persistence Forecast...")
# loading the persistence forecaste 
import os
import numpy as np
import xarray as xr

# Define the base directory containing the NetCDF files
base_dir = "/ec/res4/hpcperm/ecmv6565/fillup_uncalibrated/fillup_series"

# Define year and months to process
year = 2023
months = ["01", "02", "03"]  # January, February, March

# Initialize a list to store extracted data
all_days_data = []

# Loop through all selected months and days
for month in months:
    for day in range(1, 32):  # Max 31 days
        day_str = f"{day:02d}"  # Ensure two-digit day format
        file_path = os.path.join(base_dir, f"EUD{year}{month}{day_str}12_series.nc")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping...")
            continue  # Skip missing files
        
        # Open the NetCDF file
        with xr.open_dataset(file_path) as ds:
            # Get the last timestep discharge values
            last_discharge = ds["dis"].isel(time=-1).values  # Shape: (5695,)
            all_days_data.append(last_discharge)

# Convert the collected data to a NumPy array
persistence_array = np.array(all_days_data)  # Shape: (90, 5695)

# Broadcast to shape (90, 51, 60, 5695)
persistence_forecast = np.expand_dims(persistence_array, axis=(1, 2))  # Shape: (90, 1, 1, 5695)
persistence_forecast = np.tile(persistence_forecast, (1, 51, 60, 1))  # Final shape: (90, 51, 60, 5695)

# Save as .npy file
np.save("/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/persistence_forecast_array.npy", persistence_forecast)

print("Persistence forecast array saved successfully!")

######################################
# calculating crps for persistence forecast
#############################################

# Step 2: Define CRPS Calculation
def calculate_crps(observed, forecast_ensemble):
    """
    Compute the Continuous Ranked Probability Score (CRPS) using proper integration.
    - observed: Single observed value (float)
    - forecast_ensemble: Array of forecasted ensemble values (51,)
    - Returns: CRPS (float)
    """
    sorted_values = np.sort(np.append(forecast_ensemble, observed))
    n = len(forecast_ensemble)

    # Compute empirical CDF of forecast
    forecast_cdf = np.searchsorted(forecast_ensemble, sorted_values, side='right') / n
    observed_cdf = np.where(sorted_values < observed, 0, 1)  # Step function for observation

    # Compute integral (squared differences * interval width)
    squared_diffs = (forecast_cdf - observed_cdf) ** 2
    intervals = np.diff(sorted_values)

    return np.sum(squared_diffs[:-1] * intervals)

# Step 3: Compute CRPS for Persistence Forecast
def calculate_crps_persistence(persistence_forecast, observation_array):
    """
    Sequential CRPS computation for persistence forecast across all lead times and stations.
    """
    crps_dict = {lead_time: {} for lead_time in range(num_lead_times)}

    print("Starting CRPS calculation for Persistence Forecast...")

    for lead_time in range(num_lead_times):
        print(f"Processing Lead Time {lead_time + 1}/{num_lead_times}...")
        
        for station in range(num_stations):
            forecast_values = persistence_forecast[:, :, lead_time, station]  # Shape: (days, 51)
            observed_values = observation_array[:, lead_time, station]  # Shape: (days,)

            # Compute CRPS for valid days
            valid_mask = ~np.isnan(observed_values) & ~np.isnan(forecast_values).any(axis=1)
            crps_list = [
                calculate_crps(observed_values[day], forecast_values[day]) 
                for day in np.where(valid_mask)[0]
            ]

            crps_dict[lead_time][station] = np.nanmean(crps_list) if crps_list else np.nan

    return crps_dict

# Run the CRPS calculation for the persistence forecast
num_days = persistence_forecast.shape[0]
trimmed_observation_array = observation_array[:num_days, :, :]

# Run the CRPS calculation for the persistence forecast
crps_persistence_dict = calculate_crps_persistence(persistence_forecast, trimmed_observation_array)

# Save results
print("Saving results...")
crps_output_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/crps_persistence_results.pkl"

# Save CRPS results
with open(crps_output_path, "wb") as f:
    pickle.dump(crps_persistence_dict, f)

print(f"- CRPS results saved: {crps_output_path}")

