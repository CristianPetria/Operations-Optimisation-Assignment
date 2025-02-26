from gurobipy import Model, GRB, quicksum
import random

# # ---------------------------
# # Sets and data (example)
# # ---------------------------
# I = [1, 2, 3]      # Pharmaceutical orders
# B = [1, 2, 3, 4]   # Delivery batches
# B0 = [0, 1, 2, 3, 4]
# V = [1, 2]         # Vehicles
# N = [0, 1, 2, 3]   # Nodes: 0 = depot, others = orders
#
# # Parameters
# R = {1: 100, 2: 150, 3: 200}  # Revenue
# W = {1: 10, 2: 15, 3: 20}     # Weight
# t_lb = {1: 2, 2: 2, 3: 2}     # Lower temp bounds
# t_ub = {1: 8, 2: 8, 3: 8}     # Upper temp bounds
# CA = 30                       # Capacity
# ST_LB = {1: 8, 2: 9, 3: 7}    # For feasibility checks
# ST_UB = {1: 17, 2: 17, 3: 17}
# D = {1: 10, 2: 12, 3: 9}      # Due dates (example)
# S = {1: 16, 2: 17, 3: 15}     # Shelf lives (example)
# alpha = 1
# M = 1000000                     # Cost weight
#
# # DT[i,j]: Travel time between nodes (0 = depot, i in I)
# # Must be defined by the user. Example:
# DT = {(0,0):0, (0,1):1, (0,2):2, (0,3):1,
#       (1,0):1, (1,1):0, (1,2):1, (1,3):2,
#       (2,0):2, (2,1):1, (2,2):0, (2,3):1,
#       (3,0):1, (3,1):2, (3,2):1, (3,3):0}

def generate_random_travel_times(num_nodes, max_time=20):
    DT = {}
    for i in range(num_nodes):
        for j in range(i, num_nodes):  # Only go one way to avoid overwriting
            if i == j:
                DT[(i, j)] = 0
            else:
                time = random.randint(1, max_time)
                DT[(i, j)] = time
                DT[(j, i)] = time  # Make sure it's symmetric
    return DT



# Sets and data (extended for 20 orders)
# -------------------------------------
I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Pharmaceutical orders
B = [1, 2, 3, 4, 5]                      # Delivery batches
B0 = [0, 1, 2, 3, 4, 5]                  # Batches including depot batch 0
V = [1, 2, 3, 4, 5, 6, 7, 8]                      # Vehicles
N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # Nodes: 0 = depot, others = orders

# Parameters
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



# ---------------------------
# Model
# ---------------------------
model = Model("PharmaceuticalDelivery")

# Decision variables
a = model.addVars(I, vtype=GRB.BINARY, name="a")     # Order acceptance
x = model.addVars(I, B, vtype=GRB.BINARY, name="x")  # Assign order i to batch b
y = model.addVars(B, V, vtype=GRB.BINARY, name="y")  # Assign batch b to vehicle v
z = model.addVars(I, vtype=GRB.BINARY, name="z")     # Disposal indicator
t_b = model.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="t_b") # Batch temperature
r_b = model.addVars(N, N, B, vtype=GRB.BINARY, name="r_b") # Routing within batch
u_i = model.addVars(N, B, vtype=GRB.INTEGER, name="u_i")    # Subtour elim in routing
s = model.addVars(B0, B0, V, vtype=GRB.BINARY, name="s")      # Scheduling of batches
s_b_var = model.addVars(B, vtype=GRB.CONTINUOUS, name="s_b") # Start time of batch
c_b_var = model.addVars(B, vtype=GRB.CONTINUOUS, name="c_b") # Completion time of batch
u_bv = model.addVars(B0, V, vtype=GRB.INTEGER, name="u_bv")   # Subtour elim in scheduling
E = model.addVars(I, vtype=GRB.CONTINUOUS, name="E")         # Earliness
T = model.addVars(I, vtype=GRB.CONTINUOUS, name="T")         # Tardiness
c = model.addVars(I, vtype=GRB.CONTINUOUS, name="c")         # Completion time of orders

# Objective
DeliveryRevenue = quicksum(R[i]*a[i] for i in I)
EarlinessTardinessCost = quicksum(alpha*R[i]*(E[i] + T[i]) for i in I)
DisposalCost = quicksum(R[i]*z[i] for i in I)
model.setObjective(DeliveryRevenue - (EarlinessTardinessCost + DisposalCost), GRB.MAXIMIZE)

# Constraints

# Order assignment (9)
model.addConstrs((quicksum(x[i,b] for b in B) == a[i] for i in I), name="AssignOrderToBatch")

# Capacity (10)
model.addConstrs((quicksum(x[i,b]*W[i] for i in I) <= CA for b in B), name="Capacity")

# Temperature feasibility (11)
# If order i is in batch b, then t_lb[i] <= t_b[b] <= t_ub[i]
model.addConstrs((t_lb[i] - M*(1 - x[i,b]) <= t_b[b] for i in I for b in B), name="TempLower")
model.addConstrs((t_b[b] <= t_ub[i] + M*(1 - x[i,b]) for i in I for b in B), name="TempUpper")

# Batch to vehicle assignment (12 - 13)
model.addConstrs((quicksum(y[b,v] for v in V) <= quicksum(x[i,b] for i in I) for b in B), name="NonemptyBatchIfAssigned")
model.addConstrs((x[i,b] <= quicksum(y[b,v] for v in V) for i in I for b in B), name="LinkBatchVehicle")
model.addConstrs((quicksum(y[b,v] for v in V) <= 1 for b in B), name="OneVehiclePerBatch")

# Routing constraints (14 - 15)
# Flow balance for accepted orders
model.addConstrs((r_b[0,i,b] + quicksum(r_b[j,i,b] for j in I if j!=i) == x[i,b] for b in B for i in I), name="RoutingInflow")
model.addConstrs((r_b[i,0,b] + quicksum(r_b[i,j,b] for j in I if j!=i) == x[i,b] for b in B for i in I), name="RoutingOutflow")

# Nonempty batch: route must start and end at depot (16 - 17)
model.addConstrs((quicksum(x[i,b] for i in I) <= M * quicksum(r_b[0,j,b] for j in I) for b in B), name="StartFromDepot")
model.addConstrs((quicksum(x[i,b] for i in I) <= M * quicksum(r_b[j,0,b] for j in I) for b in B), name="ReturnToDepot")

# No loops (18)
model.addConstrs((r_b[i,i,b] == 0 for b in B for i in N), name="NoSelfLoop")

# Subtour elimination in routing (19 - 22)
model.addConstrs((u_i[0,b] == 0 for b in B), name="DepotOrderPosition")
model.addConstrs((u_i[i,b] + r_b[i,j,b] <= u_i[j,b] + M*(1 - r_b[i,j,b])
                  for b in B for i in N for j in I if i != j), name="RoutingSubtour1")
model.addConstrs((u_i[i,b] <= M * quicksum(r_b[i,j,b] for j in I if j !=i) for b in B for i in I), name="RoutingSubtour2")
model.addConstrs((u_i[i,b] <= quicksum(x[k,b] for k in I) for b in B for i in I), name="RoutingSubtour3")

# Scheduling constraints (batches on vehicles) (23 - 24)
# Flow balance in scheduling
model.addConstrs((s[0,i,v] + quicksum(s[j,i,v] for j in B if j !=i) == y[i,v] for v in V for i in B), name="SchedInflow")
model.addConstrs((s[i,0,v] + quicksum(s[i,j,v] for j in B if j !=i) == y[i,v] for v in V for i in B), name="SchedOutflow")

# Nonempty vehicle schedule start/end (25 - 26)
model.addConstrs((quicksum(y[i,v] for i in B) <= M * quicksum(s[0,j,v] for j in B) for v in V), name="SchedStart")
model.addConstrs((quicksum(y[i,v] for i in B) <= M * quicksum(s[j,0,v] for j in B) for v in V), name="SchedEnd")

# No self loops in scheduling (27)
model.addConstrs((s[b,b,v] == 0 for b in B0 for v in V), name="NoSelfLoopScheduling")

# Subtour elimination in scheduling (28 - 31)
model.addConstrs((u_bv[0, v] == 0 for v in V), name="SchedulingDepot")
model.addConstrs((u_bv[i,v] + s[i,j,v] <= u_bv[j,v] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i != j), name="SchedulingSubtour1")
model.addConstrs((u_bv[i,v] <= M * quicksum(s[i,j,v] for j in B if j != i) for v in V for i in B), name="SchedulingSubtour2")
model.addConstrs((u_bv[i, v] <= quicksum(y[j, v] for j in B) for v in V for i in B), name="SchedulingSubtour3")

# Batch timing constraints (32 - 35)
model.addConstrs((c_b_var[i] <= s_b_var[j] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i !=j), name="BatchTiming1")
model.addConstrs((s_b_var[j] <= c_b_var[i] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i !=j), name="BatchTiming2")
model.addConstrs((s_b_var[i] <= M * (1 - s[0,i,v]) for v in V for i in B), name="BatchStartFrom0")
model.addConstrs((0 <= s_b_var[i] + M * (1 - s[0,i,v]) for v in V for i in B), name="BatchStartNonNeg")

# Completion times of orders (36 - 37)
# If i precedes j in batch b
model.addConstrs((c[j] >= c[i] + DT[i,j] - M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeForward")
model.addConstrs((c[j] <= c[i] + DT[i,j] + M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeBackward")

# First visited order in batch (38 - 39)
model.addConstrs((c[i] >= s_b_var[b] + DT[0,i] - M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime1")
model.addConstrs((c[i] <= s_b_var[b] + DT[0,i] + M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime2")

# Last visited order in batch (40 - 41)
model.addConstrs((c[i] >= c_b_var[b] - M * (1 - r_b[i,0,b]) - DT[i,0] for b in B for i in I), name="LastOrderTime2")
model.addConstrs((c_b_var[b] >= c[i] + DT[i,0] - M * (1 - r_b[i,0,b]) for b in B for i in I), name="LastOrderTime1")

# Earliness and Tardiness (3 - 4)
model.addConstrs((E[i] >= D[i] - c[i] - M * (1 - a[i]) for i in I), name="EarlinessDef")
model.addConstrs((T[i] >= c[i] - D[i] - M * (1 - a[i]) for i in I), name="TardinessDef")

# Disposal (6)
model.addConstrs((c[i] - S[i] <= M * z[i] for i in I), name="DisposalDef")

# Link disposal to order rejection
model.addConstrs((z[i] >= 1 - a[i] for i in I), name="LinkDisposalLower")
model.addConstrs((z[i] <= 1 - a[i] for i in I), name="LinkDisposalUpper")

# Callback function to monitor variable 'a'
def a_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # Retrieve the objective value of the new solution
        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        print(f"\nNew feasible solution found with objective value: {obj_val}")
        
        # Retrieve the current solution for variable 'a'
        a_vars = [model.getVarByName(f"a[{i}]") for i in I]
        a_values = model.cbGetSolution(a_vars)
        
        # Identify which orders are accepted
        accepted_orders = [i for i, val in zip(I, a_values) if val > 0.5]
        rejected_orders = [i for i, val in zip(I, a_values) if val <= 0.5]
        
        print(f"Accepted orders: {accepted_orders}")
        print(f"Rejected orders: {rejected_orders}")

# Disable presolve using enumerated parameter
model.setParam(GRB.Param.Presolve, 0)

# Verify presolve is off
try:
    presolve_status = model.getParam(GRB.Param.Presolve)
    print("Presolve parameter:", presolve_status)
except AttributeError:
    print("Warning: 'getParam' method not found. Ensure gurobipy is up to date.")

# Enable Gurobi's default output and log file
model.setParam(GRB.Param.OutputFlag, 1)  # Enable Gurobi's default output
model.setParam(GRB.Param.LogFile, "gurobi_log.txt")  # Save log to a file

# Optimize with the callback
model.optimize(a_callback)

# Check for infeasibility and compute IIS if necessary
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS...")
    model.computeIIS()
    model.write("model.ilp")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"IIS Constraint: {c.constrName}")
    print("IIS written to model.ilp")

# Print final solution if optimal
if model.status == GRB.OPTIMAL:
    print("\nOptimal objective value:", model.objVal)
    print("Final accepted orders:")
    for i in I:
        if a[i].X > 0.5:
            print(f"Order {i}: Accepted")
        else:
            print(f"Order {i}: Rejected")
