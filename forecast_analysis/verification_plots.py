import numpy as np
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import random
import pandas as pd

random.seed(42)  # for reproducibility

# Load data
calibrated_forecast_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis/discharge_array.npy"
uncalibrated_forecast_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis_uncalibrated/discharge_array.npy"
observation_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis/observation_array_jan_march.npy"

station_info = pd.read_csv('/home/ecmv6565/Downloads/outlets_updated.csv')  # Load station metadata
loaded_stations = np.load('/ec/res4/hpcperm/ecmv6565/forecast_analysis/station_dict.npy', allow_pickle=True).item()

print("Loading calibrated forecast data...")
calibrated_discharge = np.load(calibrated_forecast_path)
print("Loading uncalibrated forecast data...")
uncalibrated_discharge = np.load(uncalibrated_forecast_path)
print("Loading observation data...")
observation_array = np.load(observation_path)


def plot_hydrograph(obs, fcst_cal, fcst_uncal, obs_id, ec_calib, lead_time, r_cal, beta_cal, gamma_cal, kge_cal, r_uncal, beta_uncal, gamma_uncal, kge_uncal):
    plt.figure(figsize=(12, 6))
    plt.plot(obs, label="Observation", color="black")
    plt.plot(fcst_cal, label="Calibrated Forecast", color="blue", linestyle="--")
    plt.plot(fcst_uncal, label="Uncalibrated Forecast", color="red", linestyle="--")
    plt.title(f"St: {obs_id} | EC_calib: {ec_calib} | lt: {lead_time}\n"
              f"Cal. KGE: {kge_cal:.3f}, r: {r_cal:.3f}, B: {beta_cal:.3f}, G: {gamma_cal:.3f}\n"
              f"UnCal. KGE: {kge_uncal:.3f}, r: {r_uncal:.3f}, B: {beta_uncal:.3f}, G: {gamma_uncal:.3f}")
    plt.xlabel("Time")
    plt.ylabel("Discharge")
    plt.legend()
    plt.savefig(f"./verification/hydrograph_station_{obs_id}_leadtime_{lead_time}.png")  
    plt.close()


def calculate_metrics(forecast, observation, lead_time):
    observation = np.expand_dims(observation, axis=1)  
    observation = np.repeat(observation, forecast.shape[1], axis=1)  

    results = {}

    for st in tqdm(range(forecast.shape[3]), desc=f"Calculating Metrics for Lead Time {lead_time}"):
        obs = observation[:, 0, lead_time, st]  
        fcst = forecast[:, 0, lead_time, st]  

        valid = ~np.isnan(obs) & ~np.isnan(fcst)
        obs, fcst = obs[valid], fcst[valid]

        if len(obs) > 0:
            r = np.corrcoef(fcst, obs)[0, 1] if len(obs) > 1 else 0
            beta = np.mean(fcst) / np.mean(obs) if np.mean(obs) > 0 else np.nan
            gamma = ((np.std(fcst) / np.mean(fcst)) / (np.std(obs) / np.mean(obs))) if np.mean(fcst) > 0 and np.mean(obs) > 0 else np.nan
            kge_prime = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
        else:
            kge_prime, r, beta, gamma = np.nan, np.nan, np.nan, np.nan

        results[st] = (kge_prime, r, beta, gamma)

    return results


days = 90

def run_analysis():
    print("Starting metric calculations...")

    calibrated_discharge_subset = calibrated_discharge[:days, :, :, :]
    uncalibrated_discharge_subset = uncalibrated_discharge[:days, :, :, :]
    observation_array_subset = observation_array[:days, :, :]

    for lead_time in [0]:
        print(f"\nProcessing Lead Time {lead_time}...")
        results_calibrated = calculate_metrics(calibrated_discharge_subset, observation_array_subset, lead_time)
        results_uncalibrated = calculate_metrics(uncalibrated_discharge_subset, observation_array_subset, lead_time)

        common_stations = [st for st in results_calibrated if not np.isnan(results_calibrated[st][0]) and not np.isnan(results_uncalibrated[st][0])]
        selected_stations = random.sample(common_stations, min(100, len(common_stations)))

        for st in tqdm(selected_stations, desc=f"Plotting Hydrographs for Lead Time {lead_time}"):
            obs_id = int(loaded_stations[st])
            station_row = station_info.loc[station_info['ObsID'] == obs_id]
            ec_calib = station_row['EC_calib'].values[0] if not station_row.empty else 'N/A'

            obs = observation_array_subset[:, lead_time, st]
            fcst_cal = calibrated_discharge_subset[:, 0, lead_time, st]
            fcst_uncal = uncalibrated_discharge_subset[:, 0, lead_time, st]

            kge_cal, r_cal, beta_cal, gamma_cal = results_calibrated[st]
            kge_uncal, r_uncal, beta_uncal, gamma_uncal = results_uncalibrated[st]

            plot_hydrograph(obs, fcst_cal, fcst_uncal, st, ec_calib, lead_time, r_cal, beta_cal, gamma_cal, kge_cal, r_uncal, beta_uncal, gamma_uncal, kge_uncal)


run_analysis()

print("\n🎉 Analysis complete! The plots are saved in the './verification/' directory.")

