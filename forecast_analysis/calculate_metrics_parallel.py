import numpy as np
from tqdm import tqdm
import pickle
import os

# Load data
forecast_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis/discharge_array.npy"
observation_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis/observation_array_jan_march.npy"

print("Loading forecast data...")
loaded_discharge = np.load(forecast_path)
print("Loading observation data...")
observation_array = np.load(observation_path)

def calculate_metrics(forecast, observation):
    """
    Calculates KGE' (r, beta, gamma), NSE, and RMSE for ensemble 0 using the modified KGE formula (KGE').
    """

    observation = np.expand_dims(observation, axis=1)  
    observation = np.repeat(observation, forecast.shape[1], axis=1)  

    # Initialize a new dictionary structure where the metric is the top-level key
    metrics_dict = {metric: {} for metric in ["KGE", "r", "Beta", "Gamma", "NSE", "RMSE"]}

    print(f"Calculating metrics for {forecast.shape[2]} lead times (only ensemble 0)...")

    for lt in tqdm(range(forecast.shape[2]), desc="Lead times", position=0, leave=True):
        for metric in metrics_dict:
            metrics_dict[metric][lt] = {0: {}}  # Initialize lead time and ensemble structure

        for st in range(forecast.shape[3]):
            obs = observation[:, 0, lt, st]  
            fcst = forecast[:, 0, lt, st]  

            valid = ~np.isnan(obs) & ~np.isnan(fcst)
            obs, fcst = obs[valid], fcst[valid]

            if len(obs) > 0:
                # Correlation coefficient (r)
                r = np.corrcoef(fcst, obs)[0, 1] if len(obs) > 1 else 0

                # Beta (bias ratio)
                beta = np.mean(fcst) / np.mean(obs) if np.mean(obs) > 0 else np.nan

                # Gamma (coefficient of variation ratio)
                gamma = ((np.std(fcst) / np.mean(fcst)) / (np.std(obs) / np.mean(obs))) if np.mean(fcst) > 0 and np.mean(obs) > 0 else np.nan

                # Kling-Gupta Efficiency (Modified KGE')
                kge_prime = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)

                # Nash-Sutcliffe Efficiency (NSE)
                nse = 1 - (np.sum((fcst - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)) if np.var(obs) > 0 else np.nan

                # Root Mean Square Error (RMSE)
                rmse = np.sqrt(np.mean((fcst - obs) ** 2))
            else:
                kge_prime, r, beta, gamma, nse, rmse = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Store metrics under the correct structure
            metrics_dict["KGE"][lt][0][st] = kge_prime
            metrics_dict["r"][lt][0][st] = r
            metrics_dict["Beta"][lt][0][st] = beta
            metrics_dict["Gamma"][lt][0][st] = gamma
            metrics_dict["NSE"][lt][0][st] = nse
            metrics_dict["RMSE"][lt][0][st] = rmse

        if (lt + 1) % 10 == 0:
            print(f"✅ Processed lead time {lt + 1}/{forecast.shape[2]}")

    return metrics_dict

print("Starting metric calculations for ensemble 0...")
loaded_discharge= loaded_discharge[:90, :, :, :]  # 90 to used the full 3 months 59 for jan and feb and 31 for jan 
observation_array= observation_array[:90,:, :]
metrics_dict = calculate_metrics(loaded_discharge, observation_array)

# Save results
output_path = "/ec/res4/hpcperm/ecmv6565/forecast_analysis/metrics_calculated_slurm.pkl"
with open(output_path, "wb") as f:
    pickle.dump(metrics_dict, f)

print(f"\n🎉 Calculation complete! Results saved to {output_path}")

