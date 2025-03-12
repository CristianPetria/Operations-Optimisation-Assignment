from core_model import generate_random_travel_times, generate_random_dict, generate_list, run_model
import random
import numpy as np
import matplotlib.pyplot as plt
import time

objective_values = []
number_of_accepted_orders = []
solution_time = []
optimality_gaps = []

for value in np.arange(0, 100, 1):    
    random.seed(int(value))
    settings = {}
    #batch_size

    settings["I"] = generate_list(1, 50)
    settings["B"] = generate_list(1, int(value))
    settings["B0"] = [0] + settings["B"]
    settings["V"] = generate_list(1, 10)
    settings["N"] = [0] + settings["I"]
    
    settings["R"]     = generate_random_dict(len(settings["I"]), (1000, 2000))
    settings["W"]     = generate_random_dict(len(settings["I"]), (15, 50))
    settings["t_lb"]  = generate_random_dict(len(settings["I"]), (-5, 0))
    settings["t_ub"]  = generate_random_dict(len(settings["I"]), (0, 12))
    settings["ST_LB"] = generate_random_dict(len(settings["I"]), (6, 10))
    settings["ST_UB"] = generate_random_dict(len(settings["I"]), (13, 25))
    settings["D"]     = generate_random_dict(len(settings["I"]), (20, 30))
    settings["S"]     = generate_random_dict(len(settings["I"]), (10, 20))
    settings["DT"]    = generate_random_travel_times(len(settings["N"]), max_time=40)
    
    settings["alpha"] = 1
    settings["M"]     = 1e9
    settings["CA"]    = 50
    start_time = time.time()
    result = run_model(settings, time_limit=300, presolve=0, verbose=True)
    total_time = time.time() - start_time
    optimality_gaps.append(result["gap"])
    objective_values.append(result["objective"])
    solution_time.append(total_time)
    number_of_accepted_orders.append(result["accepted"])
    print(f"Run {value}: Objective: {result['objective']}, Accepted: {result['accepted']}, Time: {total_time:.2f}s, Gap: {result['gap']:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].hist(objective_values, color='blue')
axes[0].set_xlabel("Objective value")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

axes[1].hist(number_of_accepted_orders, color='green')
axes[1].set_xlabel("Number of accepted orders")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

axes[2].hist(solution_time, color='red')
axes[2].set_xlabel("Solution time [s]")
axes[2].set_ylabel("Frequency")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("optimization_results.svg")
plt.show()


