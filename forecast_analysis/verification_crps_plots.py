# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random


def calculate_crps(observed, forecast_ensemble):
    sorted_values = np.sort(np.append(forecast_ensemble, observed))
    n = len(forecast_ensemble)
    forecast_cdf = np.searchsorted(forecast_ensemble, sorted_values, side='right') / n
    observed_cdf = np.where(sorted_values < observed, 0, 1)
    squared_diffs = (forecast_cdf - observed_cdf) ** 2
    intervals = np.diff(sorted_values)
    return np.sum(squared_diffs[:-1] * intervals)


def plot_hydrograph(calibrated_forecast, uncalibrated_forecast, observation, lead_time, station, output_path):
    days = np.arange(calibrated_forecast.shape[0])

    plt.figure(figsize=(12, 6))

    # Plot all ensemble members for calibrated and uncalibrated forecasts
    for ens in range(calibrated_forecast.shape[1]):
        plt.plot(days, calibrated_forecast[:, ens], color='blue', alpha=0.2)
        plt.plot(days, uncalibrated_forecast[:, ens], color='orange', alpha=0.2)

    # Plot observed values
    plt.plot(days, observation, color='black', label='Observed', linewidth=2)

    # Calculate CRPS for both forecasts
    crps_calibrated = np.mean([
        calculate_crps(observation[day], calibrated_forecast[day])
        for day in range(len(days)) if not np.isnan(observation[day])
    ])

    crps_uncalibrated = np.mean([
        calculate_crps(observation[day], uncalibrated_forecast[day])
        for day in range(len(days)) if not np.isnan(observation[day])
    ])

    # Plot settings
    plt.title(f'Station {station} - Lead Time {lead_time}\nCRPS Calibrated: {crps_calibrated:.3f}, CRPS Uncalibrated: {crps_uncalibrated:.3f}')
    plt.xlabel('Days')
    plt.ylabel('Discharge')
    plt.legend(['Calibrated Ensemble', 'Uncalibrated Ensemble', 'Observed'])
    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(output_path, f'station_{station}_leadtime_{lead_time}.png'))
    plt.close()


# Load data
calibrated_path = '/ec/res4/hpcperm/ecmv6565/forecast_analysis/'
uncalibrated_path = '/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/'
output_path = '/ec/res4/hpcperm/ecmv6565/forecast_analysis/verification_crps/'


print('Loading the calibrated data...')
loaded_discharge_calibrated = np.load(calibrated_path + 'discharge_array.npy')
observation_array = np.load(calibrated_path + 'observation_array_jan_march.npy')

print('Loading the uncalibrated data...')
loaded_discharge_uncalibrated = np.load(uncalibrated_path + 'discharge_array.npy')


# Filter data for lead time 59
lead_time = 15
num_stations = loaded_discharge_calibrated.shape[-1]

# Find valid stations where CRPS is not NaN for both forecasts
valid_stations = []

for station in range(num_stations):
    calibrated_forecast = loaded_discharge_calibrated[:, :, lead_time, station]
    uncalibrated_forecast = loaded_discharge_uncalibrated[:, :, lead_time, station]
    observation = observation_array[:, lead_time, station]

    if not (np.isnan(observation).all() or np.isnan(calibrated_forecast).any() or np.isnan(uncalibrated_forecast).any()):
        valid_stations.append(station)

# Randomly select 100 stations from valid ones
random.seed(42)  # Ensure reproducibility
selected_stations = random.sample(valid_stations, min(100, len(valid_stations)))

print('Generating plots...')

for station in selected_stations:
    plot_hydrograph(
        loaded_discharge_calibrated[:, :, lead_time, station],
        loaded_discharge_uncalibrated[:, :, lead_time, station],
        observation_array[:, lead_time, station],
        lead_time,
        station,
        output_path
    )

print('Plots saved successfully!')

