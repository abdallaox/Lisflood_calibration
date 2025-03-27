import numpy as np
import pickle

# Load data
print("Loading the data...")
loaded_discharge = np.load('/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/discharge_array.npy')  # Shape: (90, 51, 60, 5695)
observation_array = np.load('/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/observation_array_jan_march.npy')  # Shape: (90, 60, 5695)

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

def calculate_crps_sequential(loaded_discharge, observation_array):
    """
    Sequential CRPS computation across all lead times and stations.
    """
    num_days, num_ensembles, num_lead_times, num_stations = loaded_discharge.shape

    crps_dict = {lead_time: {} for lead_time in range(num_lead_times)}

    print("Starting CRPS calculation...")

    for lead_time in range(num_lead_times):
        print(f"Processing Lead Time {lead_time + 1}/{num_lead_times}...")
        
        for station in range(num_stations):
            forecast_values = loaded_discharge[:, :, lead_time, station]  # Shape: (days, 51)
            observed_values = observation_array[:, lead_time, station]  # Shape: (days,51)

            # Compute CRPS for valid days
            valid_mask = ~np.isnan(observed_values) & ~np.isnan(forecast_values).any(axis=1)
            crps_list = [
                calculate_crps(observed_values[day], forecast_values[day]) 
                for day in np.where(valid_mask)[0]
            ]

            crps_dict[lead_time][station] = np.nanmean(crps_list) if crps_list else np.nan

    return crps_dict

num_days = loaded_discharge.shape[0]
trimmed_observation_array = observation_array[:num_days, :, :]

# Run the CRPS calculation sequentially
crps_dict = calculate_crps_sequential(loaded_discharge, trimmed_observation_array)

# Save results
output_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/crps_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(crps_dict, f)

print("CRPS results saved to crps_results.pkl")

