import pandas as pd
import os
import numpy as np
import xarray as xr
import calendar



print('xxxxxxxx reading the forecast data into array starting xxxxxxx')
###################### 
######### loading the forecast data into array 
#######################

# User-defined settings
folder_path = "/ec/res4/scratch/ecmv6565/extracted_uncalibrated"
save_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated"
year = 2023  # Change this to the desired year
#months = [str(i).zfill(2) for i in range(1, 13)]  # Process all months (modify if needed)
months = [str(i).zfill(2) for i in range(1, 4)]  # Process all months (modify if needed)
ensembles = [f"{i:02}" for i in range(51)]  # 00 to 50
lead_times = 60
stations = 5695

print(f"Starting processing for year {year}...")

# Initialize storage
all_discharge_data = []
all_days = []

for month in months:
    days_in_month = calendar.monthrange(year, int(month))[1]
    days = [str(i).zfill(2) for i in range(1, days_in_month + 1)]
    
    print(f"\nProcessing {year}-{month} ({days_in_month} days)...")

    # Load station info from the first available file
    if not all_discharge_data:
        first_file = os.path.join(folder_path, f"EUE{year}{month}0112p00_series.nc")
        if os.path.exists(first_file):
            with xr.open_dataset(first_file) as ds:
                station_dict = {idx: str(station_id) for idx, station_id in enumerate(ds.station.values)}
            print("✔ Station information loaded.")
        else:
            print("⚠ Warning: Station file not found. Skipping station info extraction.")
            station_dict = {}

    # Initialize array for current month
    discharge_array = np.full((len(days), len(ensembles), lead_times, stations), np.nan)

    # Read NetCDF files
    for day_idx, day in enumerate(days):
        for ensemble_idx, ensemble in enumerate(ensembles):
            filename = f"EUE{year}{month}{day}12p{ensemble}_series.nc"
            file_path = os.path.join(folder_path, filename)

            if os.path.exists(file_path):
                with xr.open_dataset(file_path) as ds:
                    discharge_array[day_idx, ensemble_idx, :, :] = ds["dis"].values
            else:
                print(f"  ⚠ Missing file: {filename}")

        print(f"  ✅ Processed {year}-{month}-{day}")

    print(f"✔ Completed {year}-{month}. Shape: {discharge_array.shape}")
    
    all_discharge_data.append(discharge_array)
    all_days.extend(days)

# Combine all months
final_discharge_array = np.concatenate(all_discharge_data, axis=0)

# Save results
np.save(os.path.join(save_path, 'discharge_array.npy'), final_discharge_array)
np.save(os.path.join(save_path, 'station_dict.npy'), station_dict)

print("\n🎉 Processing complete!")
print(f"Final data shape: {final_discharge_array.shape}")
print(f"Data saved in: {save_path}")


###################### 
######### loading the observation data into array 
#######################

print('reading the observation data csv')
# Path to the CSV file
file_path = "/ec/vol/efas/discharge_obs/6.0/lsf_calib/Qobs_06.csv"
#station_dict = np.load('/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/station_dict.npy', allow_pickle=True).item() ### stations indexes 

# Read the CSV file
data = pd.read_csv(file_path)


print('xxxxxxxx reading the obs data into array starting xxxxxxx')
# Convert observation dataframe timestamps to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Generate a time index for January and february 2023 (6-hourly starting at 18:00)
time_index = pd.date_range(start='2023-01-01 18:00', end='2023-04-28 18:00', freq='6H')

# Initialize the observation array with NaN values
observation_array = np.full((90, 60, 5695), np.nan)

# Iterate through each station in the dictionary
for forecast_idx, station_id in station_dict.items():
    # Check if the station ID exists in the observation dataframe
    if str(station_id) in data.columns:
        # Extract the observation data for the station
        station_obs = data[['Timestamp', str(station_id)]]
        station_obs.set_index('Timestamp', inplace=True)
        
        # Align with the forecast time index
        station_obs = station_obs.reindex(time_index)
        
        # Fill the observation array for each day
        for day_idx in range(observation_array.shape[0]):
            start_time = day_idx * 24 // 6 + 3  # Start at 18:00
            end_time = start_time + 60
            
            # Extract the data slice and ensure it has 60 elements
            data_slice = station_obs.iloc[start_time:end_time].values.flatten()
            if len(data_slice) < 60:
                # Pad with NaN if slice is shorter than 60
                padded_slice = np.full(60, np.nan)
                padded_slice[:len(data_slice)] = data_slice
            else:
                # Truncate to 60 if slice is longer than 60
                padded_slice = data_slice[:60]
            
            # Assign to observation array
            observation_array[day_idx, :, forecast_idx] = padded_slice

# Final observation array has the shape (90, 60, 5695)
print(observation_array.shape)
# Define the save path
save_path = '/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated'
file_name = 'observation_array_jan_march.npy'
full_path = os.path.join(save_path, file_name)

# Save the NumPy array
try:
    np.save(full_path, observation_array)
    print(f"Array saved successfully to {full_path}")
except Exception as e:
    print(f"An error occurred while saving the array: {e}")



