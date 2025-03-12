import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from core_model import run_model, generate_random_travel_times
import random
import seaborn as sns

def create_random_settings(seed):
    random.seed(seed)
    num_orders = 20
    I = list(range(1, num_orders + 1))
    B = list(range(1, 6))
    B0 = [0] + B
    V = list(range(1, 9))
    N = [0] + I
    R = {i: random.randint(1000, 2000) for i in I}  #
    W = {i: random.randint(5, 25) for i in I}       
    t_lb = {i: random.randint(1, 5) for i in I}     # Lower temp bounds
    t_ub = {i: random.randint(6, 15) for i in I}    # Upper temp bounds
    ST_LB = {i: random.randint(6, 10) for i in I}   # Start time lower bounds
    ST_UB = {i: random.randint(15, 20) for i in I}  # Start time upper bounds
    D = {i: random.randint(5, 15) for i in I}       # Due dates
    S = {i: random.randint(15, 20) for i in I}      # Shelf lives
    DT = generate_random_travel_times(len(N), max_time=20)
    settings = {
        "I": I,
        "B": B,
        "B0": B0,
        "V": V,
        "N": N,
        "R": R,
        "W": W,
        "t_lb": t_lb,
        "t_ub": t_ub,
        "ST_LB": ST_LB,
        "ST_UB": ST_UB,
        "D": D,
        "S": S,
        "DT": DT,
        "alpha": 1,
        "M": 1e7,
        "CA": 25
    }
    return settings

def run_stochastic_analysis(n_runs=30, time_limit=3000, output_dir="stochastic_results", start_seed=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    start_time = time.time()
    
    for i in range(n_runs):
        seed = i + int(start_seed)
        settings = create_random_settings(seed)
        model_result = run_model(settings, time_limit, presolve=0, verbose=False)
        results.append({
            "Run": i + 1,
            "Seed": seed,
            "Objective": model_result["objective"],
            "Accepted": model_result["accepted"],
            "Rejected": model_result["rejected"],
            "SolveTime": model_result["solve_time"],
            "Gap": model_result["gap"]
        })
        
        print(f"Run {i+1} completed. Objective: {model_result['objective']}, "
              f"Accepted: {model_result['accepted']}, Time: {model_result['solve_time']:.2f}s")
    
    total_time = time.time() - start_time
    
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{output_dir}/stochastic_analysis_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    stats = {
        "Mean": results_df["Objective"].mean(),
        "Median": results_df["Objective"].median(),
        "StdDev": results_df["Objective"].std(),
        "Min": results_df["Objective"].min(),
        "Max": results_df["Objective"].max(),
        "Range": results_df["Objective"].max() - results_df["Objective"].min(),
        "Q1": results_df["Objective"].quantile(0.25),
        "Q3": results_df["Objective"].quantile(0.75),
        "IQR": results_df["Objective"].quantile(0.75) - results_df["Objective"].quantile(0.25)
    }
    return results_df, stats
def cumulative_mean(data_array):
    #recursive function to calculate cumulative mean
    if len(data_array) == 0:
        return []
    else:
        cumulative_means = [data_array[0]]
        for i in range(1, len(data_array)):
            cumulative_means.append((cumulative_means[i-1] * i + data_array[i]) / (i + 1))
        return cumulative_means


if __name__ == "__main__":
    sns.set_style("whitegrid")
    means = []
    stddevs = []
    medians = []
    vals = np.arange(0, 1000, 5)
    for n in vals:
        results_df, stats = run_stochastic_analysis(n_runs=5, time_limit=300, start_seed=n)
        means.append(stats["Mean"])
        stddevs.append(stats["StdDev"])
        medians.append(stats["Median"])
            
    means = np.array(means)
    stddevs = np.array(stddevs)
    medians = np.array(medians)
    avg_means = []
    avg_stds = []
    avg_medians = []
    for i in range(len(means)):
        avg_means.append(np.mean(means[:i+1]))
        avg_stds.append(np.std(means[:i+1]))
        avg_medians.append(np.median(means[:i+1]))


    plt.figure(figsize=(10, 6))
    plt.plot(vals, avg_means, label='Mean', color='blue', marker='o')
    plt.xlabel('number of runs')
    plt.ylabel('Average objective value')
    plt.savefig('mean.png', dpi=300, bbox_inches='tight')
    plt.savefig('mean.svg', format='svg', bbox_inches='tight')
    plt.show()