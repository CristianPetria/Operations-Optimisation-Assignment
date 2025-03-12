import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import random
from datetime import datetime
from core_model import run_model, generate_list, generate_random_travel_times

MIP_GAP_TOLERANCE = 0.001

def create_random_settings(seed, parameter_name=None, param_value=None):
    random.seed(seed)
    num_orders = 20
    I = list(range(1, num_orders + 1))
    B = list(range(1, 6))
    B0 = [0] + B
    V = list(range(1, 9))
    N = [0] + I
    
    R = {i: random.randint(1000, 2000) for i in I}  
    W = {i: random.randint(5, 25) for i in I}       
    t_lb = {i: random.randint(1, 5) for i in I}     
    t_ub = {i: random.randint(6, 15) for i in I}    
    ST_LB = {i: random.randint(6, 10) for i in I}   
    ST_UB = {i: random.randint(15, 20) for i in I}  
    D = {i: random.randint(5, 15) for i in I}       
    S = {i: random.randint(15, 20) for i in I}      
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
    if parameter_name and param_value is not None:
        if parameter_name == "CA":  
            settings["CA"] = param_value
        elif parameter_name == "alpha":  
            settings["alpha"] = param_value
        elif parameter_name == "num_vehicles":  
            settings["V"] = generate_list(1, param_value)
        elif parameter_name == "num_batches":  
            settings["B"] = generate_list(1, param_value)
            settings["B0"] = [0] + settings["B"]
    return settings

def run_sensitivity_analysis(parameter_name, parameter_values, time_limit=300, 
                             output_dir="results", presolve=0, verbose=True, num_runs=200):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_results = []
    avg_results = []
    base_seed = 42
    for param_value in parameter_values:
        param_runs = []
        for run in range(num_runs):
            seed = base_seed + run
            
            random_settings = create_random_settings(seed, parameter_name, param_value)
            
            run_verbose = verbose and run == 0
            
            model_result = run_model(random_settings, time_limit, presolve, verbose=run_verbose)
            
            param_runs.append({
                "Parameter": param_value,
                "Run": run + 1,
                "Seed": seed,
                "Objective": model_result["objective"],
                "Accepted": model_result["accepted"],
                "Rejected": model_result["rejected"],
                "SolveTime": model_result["solve_time"],
                "Gap": model_result["gap"]
            })
            
        runs_df = pd.DataFrame(param_runs)   
        avg_objective = runs_df["Objective"].mean()
        avg_accepted = runs_df["Accepted"].mean()
        avg_rejected = runs_df["Rejected"].mean()
        avg_solve_time = runs_df["SolveTime"].mean()
        avg_gap = runs_df["Gap"].mean()
        std_objective = runs_df["Objective"].std()
        avg_results.append({
            "Parameter": param_value,
            "Objective": avg_objective,
            "Accepted": avg_accepted,
            "Rejected": avg_rejected,
            "SolveTime": avg_solve_time,
            "StdObjective": std_objective,
            "Gap": avg_gap
        })
        
        all_results.extend(param_runs)
    
    results_df = pd.DataFrame(avg_results)
    all_runs_df = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{output_dir}/sensitivity_{parameter_name}_{timestamp}.csv"
    all_runs_path = f"{output_dir}/sensitivity_{parameter_name}_all_runs_{timestamp}.csv"
    
    results_df.to_csv(csv_path, index=False)
    all_runs_df.to_csv(all_runs_path, index=False)
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))
    
    axs[0].errorbar(results_df["Parameter"], results_df["Objective"], 
                   yerr=results_df["StdObjective"], fmt='o-', 
                   color="royalblue", linewidth=2, capsize=5)
    axs[0].set_xlabel(parameter_name, fontsize=12)
    axs[0].set_ylabel("Objective value", fontsize=12)
    axs[0].grid(True, alpha=0.3)
    for x, y in zip(results_df["Parameter"], results_df["Objective"]):
        if not np.isnan(y):
            axs[0].annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 10),
                            textcoords='offset points', ha='center')
    
    axs[1].errorbar(results_df["Parameter"], results_df["Accepted"],
                   yerr=results_df["Accepted"].std(), fmt='o-', 
                   color="green", linewidth=2, capsize=5)
    axs[1].set_xlabel(parameter_name, fontsize=12)
    axs[1].set_ylabel("Number of accepted orders", fontsize=12)
    axs[1].grid(True, alpha=0.3)
    for x, y in zip(results_df["Parameter"], results_df["Accepted"]):
        axs[1].annotate(f'{y:.1f}', xy=(x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center')
    
    axs[2].errorbar(results_df["Parameter"], results_df["SolveTime"],
                   yerr=results_df["SolveTime"].std(), fmt='o-', 
                   color="red", linewidth=2, capsize=5)
    axs[2].set_xlabel(parameter_name, fontsize=12)
    axs[2].set_ylabel("time [s]", fontsize=12)
    axs[2].grid(True, alpha=0.3)
    for x, y in zip(results_df["Parameter"], results_df["SolveTime"]):
        axs[2].annotate(f'{y:.1f}', xy=(x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center')
    
    # Add optimality gap plot
    axs[3].errorbar(results_df["Parameter"], results_df["Gap"] * 100,  # Convert to percentage
                   yerr=results_df["Gap"].std() * 100, fmt='o-', 
                   color="orange", linewidth=2, capsize=5)
    axs[3].set_xlabel(parameter_name, fontsize=12)
    axs[3].set_ylabel("optimality gap [%]", fontsize=12)
    axs[3].grid(True, alpha=0.3)
    for x, y in zip(results_df["Parameter"], results_df["Gap"] * 100):  # Convert to percentage
        axs[3].annotate(f'{y:.2f}%', xy=(x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plot_path = f"{output_dir}/sensitivity_{parameter_name}_analysis_{timestamp}.png"
    svg_path = f"{output_dir}/sensitivity_{parameter_name}_analysis_{timestamp}.svg"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, dpi=300, bbox_inches='tight')  
    return results_df

def run_paper_style_sensitivity():
    output_dir = "sensitivity_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    time_limit = 300
    
    capacity_values = [10, 15, 20, 25, 30, 35, 40]   
    alpha_values    = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  
    vehicle_counts  = [2, 4, 6, 8, 10, 12, 14]        
    batch_counts    = [2, 3, 4, 5, 6, 7, 8]        
    
    capacity_results = run_sensitivity_analysis("CA", capacity_values, time_limit, output_dir)
    
    alpha_results = run_sensitivity_analysis("alpha", alpha_values, time_limit, output_dir)
    
    vehicle_results = run_sensitivity_analysis("num_vehicles", vehicle_counts, time_limit, output_dir)
    
    batch_results = run_sensitivity_analysis("num_batches", batch_counts, time_limit, output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    axs[0, 0].errorbar(capacity_results["Parameter"], capacity_results["Objective"],
                      yerr=capacity_results["StdObjective"], fmt='o-', capsize=5)
    axs[0, 0].set_xlabel("Vehicle capacity", fontsize=12)
    axs[0, 0].set_ylabel("Objective value", fontsize=12)
    axs[0, 0].grid(True)
    
    axs[0, 1].errorbar(alpha_results["Parameter"], alpha_results["Objective"],
                      yerr=alpha_results["StdObjective"], fmt='o-', capsize=5)
    axs[0, 1].set_xlabel("Alpha", fontsize=12)
    axs[0, 1].set_ylabel("Objective value", fontsize=12)
    axs[0, 1].grid(True)
    
    axs[1, 0].errorbar(vehicle_results["Parameter"], vehicle_results["Objective"],
                      yerr=vehicle_results["StdObjective"], fmt='o-', capsize=5)
    axs[1, 0].set_xlabel("Number of vehicles", fontsize=12)
    axs[1, 0].set_ylabel("Objective value", fontsize=12)
    axs[1, 0].grid(True)
    
    axs[1, 1].errorbar(batch_results["Parameter"], batch_results["Objective"],
                      yerr=batch_results["StdObjective"], fmt='o-', capsize=5)
    axs[1, 1].set_xlabel("Number of batches", fontsize=12)
    axs[1, 1].set_ylabel("Objective value", fontsize=12)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    combined_path = f"{output_dir}/combined_sensitivity_analysis_{timestamp}.png"
    svg_all = f"{output_dir}/combined_sensitivity_analysis_{timestamp}.svg"
    plt.savefig(svg_all, dpi=300, bbox_inches='tight')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined objective plot saved to {combined_path}")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].plot(capacity_results["Parameter"], capacity_results["Gap"] * 100, 'o-', color="orange")
    axs[0, 0].set_xlabel("Vehicle capacity", fontsize=12)
    axs[0, 0].set_ylabel("Optimality gap [%]", fontsize=12)
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(alpha_results["Parameter"], alpha_results["Gap"] * 100, 'o-', color="orange")
    axs[0, 1].set_xlabel("Alpha", fontsize=12)
    axs[0, 1].set_ylabel("Optimality gap [%]", fontsize=12)
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(vehicle_results["Parameter"], vehicle_results["Gap"] * 100, 'o-', color="orange")
    axs[1, 0].set_xlabel("Number of vehicles", fontsize=12)
    axs[1, 0].set_ylabel("Optimality gap [%]", fontsize=12)
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(batch_results["Parameter"], batch_results["Gap"] * 100, 'o-', color="orange")
    axs[1, 1].set_xlabel("Number of batches", fontsize=12)
    axs[1, 1].set_ylabel("Optimality gap [%]", fontsize=12)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    gap_plot_path = f"{output_dir}/combined_gap_analysis_{timestamp}.png"
    gap_svg_all = f"{output_dir}/combined_gap_analysis_{timestamp}.svg"
    plt.savefig(gap_svg_all, dpi=300, bbox_inches='tight')
    plt.savefig(gap_plot_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    start_time = time.time()
    run_paper_style_sensitivity()
    total_time = time.time() - start_time
