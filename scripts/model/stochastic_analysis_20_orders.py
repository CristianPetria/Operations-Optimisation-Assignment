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

def run_stochastic_analysis(n_runs=30, time_limit=3000, output_dir="stochastic_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    start_time = time.time()
    
    for i in range(n_runs):
        seed = i + 42
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
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df["Objective"], kde=True, color='royalblue')
    plt.axvline(stats["Mean"], color='red', linestyle='--', label=f'Mean: {stats["Mean"]:.2f}')
    plt.axvline(stats["Median"], color='green', linestyle='--', label=f'Median: {stats["Median"]:.2f}')
    plt.xlabel("Objective value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    hist_path = f"{output_dir}/objective_histogram_{timestamp}.png"
    hist_svg = f"{output_dir}/objective_histogram_{timestamp}.svg"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.savefig(hist_svg, format='svg', bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=results_df["Objective"], color='royalblue')
    plt.ylabel("Objective value", fontsize=12)
    plt.grid(alpha=0.3)
    
    box_path = f"{output_dir}/objective_boxplot_{timestamp}.png"
    box_svg = f"{output_dir}/objective_boxplot_{timestamp}.svg"
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.savefig(box_svg, format='svg', bbox_inches='tight')
    
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["Run"], results_df["Objective"], marker='o', linestyle='-', color='royalblue')
    plt.axhline(stats["Mean"], color='red', linestyle='--', label=f'Mean: {stats["Mean"]:.2f}')
    plt.fill_between(results_df["Run"], 
                     stats["Mean"] - stats["StdDev"], 
                     stats["Mean"] + stats["StdDev"], 
                     color='red', alpha=0.2, label=f'±1 StdDev: {stats["StdDev"]:.2f}')
    plt.xlabel("Run number", fontsize=12)
    plt.ylabel("Objective value", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    runs_path = f"{output_dir}/objective_runs_{timestamp}.png"
    runs_svg = f"{output_dir}/objective_runs_{timestamp}.svg"
    plt.savefig(runs_path, dpi=300, bbox_inches='tight')
    plt.savefig(runs_svg, format='svg', bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df["SolveTime"], results_df["Objective"], c=results_df["Accepted"], 
                cmap='viridis', alpha=0.8)
    plt.colorbar(label="Number of accepted orders")
    plt.xlabel("Solve time [s]", fontsize=12)
    plt.ylabel("Objective value", fontsize=12)
    plt.grid(alpha=0.3)
    
    scatter_path = f"{output_dir}/objective_vs_time_{timestamp}.png"
    scatter_svg = f"{output_dir}/objective_vs_time_{timestamp}.svg"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.savefig(scatter_svg, format='svg', bbox_inches='tight')
    
    print(f"\nStochastic analysis completed in {total_time:.2f} seconds")
    print(f"Results saved to {csv_path}")
    print(f"Plots saved to {output_dir}/")
    
    return results_df, stats

if __name__ == "__main__":
    sns.set_style("whitegrid")
    vals = np.arange(10, 200, 10)
    for n in vals:
        results_df, stats = run_stochastic_analysis(n_runs=n, time_limit=30000)z