import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
from core_model import run_model, generate_list
from sensitivity_analysis_20 import create_random_settings

MIP_GAP_TOLERANCE = 0.001  

def create_base_settings_for_surface():
    return create_random_settings(seed=42)

def create_settings_for_combination(base_settings, alpha, num_vehicles):
    settings = base_settings.copy()
    settings["alpha"] = alpha
    settings["V"] = generate_list(1, num_vehicles)
    return settings

def run_multi_seed_combination(base_settings, alpha, num_vehicles, num_runs=40, time_limit=300, presolve=0, verbose=False):
    all_results = []
    base_seed = 42
    
    for run in range(num_runs):
        seed = base_seed + run
        random_settings = create_random_settings(seed, parameter_name=None, param_value=None)
        random_settings["alpha"] = alpha
        random_settings["V"] = generate_list(1, num_vehicles)
        
        run_verbose = verbose and run == 0
        model_result = run_model(random_settings, time_limit, presolve, verbose=run_verbose)
        
        all_results.append({
            "Alpha": alpha,
            "Vehicles": num_vehicles,
            "Run": run + 1,
            "Seed": seed,
            "Objective": model_result["objective"],
            "Accepted": model_result["accepted"],
            "Rejected": model_result["rejected"],
            "SolveTime": model_result["solve_time"],
            "Gap": model_result["gap"]
        })
    
    # Calculate mean values and standard deviations
    runs_df = pd.DataFrame(all_results)
    summary = {
        "Alpha": alpha,
        "Vehicles": num_vehicles,
        "Objective": runs_df["Objective"].mean(),
        "ObjectiveStd": runs_df["Objective"].std(),
        "Accepted": runs_df["Accepted"].mean(),
        "AcceptedStd": runs_df["Accepted"].std(),
        "SolveTime": runs_df["SolveTime"].mean(),
        "SolveTimeStd": runs_df["SolveTime"].std(),
        "Gap": runs_df["Gap"].mean(),
        "GapStd": runs_df["Gap"].std()
    }
    return summary, all_results

def run_grid_sensitivity_analysis(alpha_values, vehicle_counts, time_limit=300, output_dir="grid_results", presolve=0, verbose=True, num_runs=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_settings = create_base_settings_for_surface()
    summaries = []
    all_results = []
    total_combinations = len(alpha_values) * len(vehicle_counts)
    counter = 1
    for alpha in alpha_values:
        for num_vehicles in vehicle_counts:
            print(f"\nRunning combination {counter}/{total_combinations}: alpha = {alpha}, vehicles = {num_vehicles}")
            print(f"Running {num_runs} simulations per combination...")
            
            summary, runs = run_multi_seed_combination(base_settings, alpha, num_vehicles, num_runs, time_limit, presolve, verbose)
            
            summaries.append(summary)
            all_results.extend(runs)
            
            print(f"  Average objective: {summary['Objective']:.2f} ± {summary['ObjectiveStd']:.2f}")
            print(f"  Average accepted orders: {summary['Accepted']:.2f} ± {summary['AcceptedStd']:.2f}")
            
            counter += 1
    
    # Create DataFrames
    summary_df = pd.DataFrame(summaries)
    all_runs_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"{output_dir}/grid_sensitivity_summary_{num_runs}runs_{timestamp}.csv"
    all_runs_path = f"{output_dir}/grid_sensitivity_all_runs_{num_runs}runs_{timestamp}.csv"
    
    summary_df.to_csv(summary_path, index=False)
    all_runs_df.to_csv(all_runs_path, index=False)
    
    print(f"Results saved to {summary_path}")
    
    objective_pivot = summary_df.pivot(index="Alpha", columns="Vehicles", values="Objective")
    std_pivot = summary_df.pivot(index="Alpha", columns="Vehicles", values="ObjectiveStd")
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    heatmap = plt.pcolor(objective_pivot.columns, objective_pivot.index, objective_pivot, cmap='viridis')
    plt.colorbar(heatmap, label=f"mean objective value ({num_runs} runs)")
    plt.xlabel("number of vehicles", fontsize=14)
    plt.ylabel("alpha", fontsize=14)    
    for i in range(len(objective_pivot.index)):
        for j in range(len(objective_pivot.columns)):
            mean_val = objective_pivot.iloc[i, j]
            std_val = std_pivot.iloc[i, j]
            if not np.isnan(mean_val):
                plt.text(objective_pivot.columns[j], objective_pivot.index[i], f"{mean_val:.0f}\n±{std_val:.0f}", ha="center", va="center", color="white", fontweight="bold",fontsize=9)
    heatmap_path = f"{output_dir}/grid_heatmap_{num_runs}runs_{timestamp}.png"
    heatmap_svg = f"{output_dir}/grid_heatmap_{num_runs}runs_{timestamp}.svg"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.savefig(heatmap_svg, format='svg', bbox_inches='tight')
    return summary_df, all_runs_df, objective_pivot

if __name__ == "__main__":
    alpha_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    vehicle_counts = [2, 4, 6, 8, 10, 12, 14]
    num_runs = 100
    time_limit = 120  
    summary_df, all_runs_df, pivot_table = run_grid_sensitivity_analysis(alpha_values, vehicle_counts, time_limit, num_runs=num_runs, verbose=False)
    