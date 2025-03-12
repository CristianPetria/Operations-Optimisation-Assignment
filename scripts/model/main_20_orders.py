from gurobipy import GRB
import random
from core_model import generate_random_travel_times, create_optimization_model

random.seed(1)

I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Pharmaceutical orders
B = [1, 2, 3, 4, 5]                      # Delivery batches
B0 = [0, 1, 2, 3, 4, 5]                  # Batches including depot batch 0
V = [1, 2, 3, 4, 5, 6, 7, 8]                      # Vehicles
N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # Nodes: 0 = depot, others = orders

R = {1: 1000, 2: 1500, 3: 2000, 4: 1200, 5: 1800, 6: 1400, 7: 1600, 8: 1100, 9: 1900, 10: 1300,
     11: 1700, 12: 1250, 13: 1800, 14: 1400, 15: 1550, 16: 1350, 17: 1450, 18: 1650, 19: 1500, 20: 1750}  # Revenue
W = {1: 10, 2: 15, 3: 20, 4: 25, 5: 18, 6: 14, 7: 8, 8: 22, 9: 5, 10: 12,
     11: 9, 12: 13, 13: 11, 14: 16, 15: 14, 16: 7, 17: 6, 18: 10, 19: 19, 20: 17}                         # Weight
t_lb = {1: 2, 2: 4, 3: 3, 4: 5, 5: 6, 6: 1, 7: 3, 8: 2, 9: 1, 10: 4,
        11: 3, 12: 2, 13: 5, 14: 1, 15: 4, 16: 2, 17: 3, 18: 4, 19: 1, 20: 3}                             # Lower temp bounds
t_ub = {1: 8, 2: 9, 3: 7, 4: 10, 5: 11, 6: 8, 7: 6, 8: 12, 9: 9, 10: 10,
        11: 13, 12: 11, 13: 14, 14: 12, 15: 9, 16: 10, 17: 12, 18: 13, 19: 8, 20: 11}                     # Upper temp bounds
CA = 25                                                                                                   # Capacity
ST_LB = {1: 8, 2: 9, 3: 7, 4: 10, 5: 8, 6: 9, 7: 6, 8: 8, 9: 7, 10: 9,
         11: 6, 12: 8, 13: 9, 14: 7, 15: 8, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10}                           # Feasibility checks
ST_UB = {1: 17, 2: 16, 3: 17, 4: 18, 5: 16, 6: 17, 7: 15, 8: 18, 9: 17, 10: 16,
         11: 19, 12: 18, 13: 20, 14: 17, 15: 16, 16: 15, 17: 18, 18: 17, 19: 16, 20: 19}
D = {1: 10, 2: 12, 3: 9, 4: 11, 5: 15, 6: 13, 7: 10, 8: 14, 9: 8, 10: 12,
     11: 11, 12: 13, 13: 9, 14: 10, 15: 14, 16: 8, 17: 9, 18: 11, 19: 13, 20: 10}                         # Due dates
S = {1: 16, 2: 17, 3: 15, 4: 16, 5: 18, 6: 17, 7: 14, 8: 19, 9: 15, 10: 16,
     11: 18, 12: 17, 13: 19, 14: 20, 15: 16, 16: 17, 17: 18, 18: 15, 19: 19, 20: 18}                      # Shelf lives
alpha = 1                                                                                                 # Coefficient
M = 1e7                                                                                                   # Cost weight

# DT[i, j]: Travel time between nodes (0 = depot, i in I)
DT = generate_random_travel_times(len(N), max_time=20)
# Create settings dictionary to pass to the model
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
    "alpha": alpha,
    "M": M,
    "CA": CA
}
model, variables = create_optimization_model(settings)
a = variables["a"]
def a_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        a_vars = [model.getVarByName(f"a[{i}]") for i in I]
        a_values = model.cbGetSolution(a_vars)  
        accepted_orders = [i for i, val in zip(I, a_values) if val > 0.5]
        rejected_orders = [i for i, val in zip(I, a_values) if val <= 0.5]
model.setParam(GRB.Param.Presolve, 0)  # Disable presolve

model.setParam(GRB.Param.OutputFlag, 1)  
model.setParam(GRB.Param.LogFile, "gurobi_log.txt")  
model._I = I
model.optimize(a_callback)
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS...")
    model.computeIIS()
    model.write("model.ilp")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"IIS Constraint: {c.constrName}")
    print("IIS written to model.ilp")
if model.status == GRB.OPTIMAL:
    print("\nOptimal objective value:", model.objVal)
    print("Final accepted orders:")
    for i in I:
        if a[i].X > 0.5:
            print(f"Order {i}: Accepted")
        else:
            print(f"Order {i}: Rejected")