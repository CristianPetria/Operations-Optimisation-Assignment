from gurobipy import Model, GRB, quicksum

# ---------------------------
# Sets and data (example)
# ---------------------------
I = [1, 2, 3]      # Pharmaceutical orders
B = [1, 2, 3, 4]   # Delivery batches
B0 = [0, 1, 2, 3, 4]
V = [1, 2]         # Vehicles
N = [0, 1, 2, 3]   # Nodes: 0 = depot, others = orders

# Parameters
R = {1: 100, 2: 150, 3: 200}  # Revenue
W = {1: 10, 2: 15, 3: 20}     # Weight
t_lb = {1: 2, 2: 2, 3: 2}     # Lower temp bounds
t_ub = {1: 8, 2: 8, 3: 8}     # Upper temp bounds
CA = 30                       # Capacity
ST_LB = {1: 8, 2: 9, 3: 7}    # For feasibility checks
ST_UB = {1: 17, 2: 17, 3: 17} 
D = {1: 10, 2: 12, 3: 9}      # Due dates (example)
S = {1: 16, 2: 17, 3: 15}     # Shelf lives (example)
alpha = 1   
M = 1000000                     # Cost weight

# DT[i,j]: Travel time between nodes (0 = depot, i in I)
# Must be defined by the user. Example:
DT = {(0,0):0, (0,1):1, (0,2):2, (0,3):1,
      (1,0):1, (1,1):0, (1,2):1, (1,3):2,
      (2,0):2, (2,1):1, (2,2):0, (2,3):1,
      (3,0):1, (3,1):2, (3,2):1, (3,3):0}

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

# Order assignment
model.addConstrs((quicksum(x[i,b] for b in B) == a[i] for i in I), name="AssignOrderToBatch")

# Capacity
model.addConstrs((quicksum(x[i,b]*W[i] for i in I) <= CA for b in B), name="Capacity")

# Temperature feasibility
# If order i is in batch b, then t_lb[i] <= t_b[b] <= t_ub[i]
model.addConstrs((t_lb[i] - M*(1 - x[i,b]) <= t_b[b] for i in I for b in B), name="TempLower")
model.addConstrs((t_b[b] <= t_ub[i] + M*(1 - x[i,b]) for i in I for b in B), name="TempUpper")

# Batch to vehicle assignment
model.addConstrs((quicksum(y[b,v] for v in V) <= quicksum(x[i,b] for i in I) for b in B), name="NonemptyBatchIfAssigned")
model.addConstrs((x[i,b] <= quicksum(y[b,v] for v in V) for i in I for b in B), name="LinkBatchVehicle")
model.addConstrs((quicksum(y[b,v] for v in V) <= 1 for b in B), name="OneVehiclePerBatch")

# Routing constraints
# Flow balance for accepted orders
model.addConstrs((r_b[0,i,b] + quicksum(r_b[j,i,b] for j in I if j!=i) == x[i,b] for b in B for i in I), name="RoutingInflow")
model.addConstrs((r_b[i,0,b] + quicksum(r_b[i,j,b] for j in I if j!=i) == x[i,b] for b in B for i in I), name="RoutingOutflow")

# Nonempty batch: route must start and end at depot
model.addConstrs((quicksum(x[i,b] for i in I) <= M * quicksum(r_b[0,j,b] for j in I) for b in B), name="StartFromDepot")
model.addConstrs((quicksum(x[i,b] for i in I) <= M * quicksum(r_b[j,0,b] for j in I) for b in B), name="ReturnToDepot")

# No loops
model.addConstrs((r_b[i,i,b] == 0 for b in B for i in N), name="NoSelfLoop")

# Subtour elimination in routing
model.addConstrs((u_i[0,b] == 0 for b in B), name="DepotOrderPosition")
model.addConstrs((u_i[i,b] + r_b[i,j,b] <= u_i[j,b] + M*(1 - r_b[i,j,b]) 
                  for b in B for i in N for j in I if i != j), name="RoutingSubtour1")
model.addConstrs((u_i[i,b] <= M * quicksum(r_b[i,j,b] for j in I if j !=i) for b in B for i in I), name="RoutingSubtour2")
model.addConstrs((u_i[i,b] <= quicksum(x[k,b] for k in I) for b in B for i in I), name="RoutingSubtour3")

# Scheduling constraints (batches on vehicles)
# Flow balance in scheduling
model.addConstrs((s[0,i,v] + quicksum(s[j,i,v] for j in B if j !=i) == y[i,v] for v in V for i in B), name="SchedInflow")
model.addConstrs((s[i,0,v] + quicksum(s[i,j,v] for j in B if j !=i) == y[i,v] for v in V for i in B), name="SchedOutflow")

# Nonempty vehicle schedule start/end
model.addConstrs((quicksum(y[i,v] for i in B) <= M * quicksum(s[0,j,v] for j in B) for v in V), name="SchedStart")
model.addConstrs((quicksum(y[i,v] for i in B) <= M * quicksum(s[j,0,v] for j in B) for v in V), name="SchedEnd")

# No self loops in scheduling
model.addConstrs((s[b,b,v] == 0 for b in B0 for v in V), name="NoSelfLoopScheduling")

# Subtour elimination in scheduling
model.addConstrs((u_bv[0, v] == 0 for v in V), name="SchedulingDepot")
model.addConstrs((u_bv[i,v] + s[i,j,v] <= u_bv[j,v] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i != j), name="SchedulingSubtour1")
model.addConstrs((u_bv[i,v] <= M * quicksum(s[i,j,v] for j in B if j != i) for v in V for i in B), name="SchedulingSubtour2")
model.addConstrs((u_bv[i, v] <= quicksum(y[j, v] for j in B) for v in V for i in B), name="SchedulingSubtour3")

# Batch timing constraints
model.addConstrs((c_b_var[i] <= s_b_var[j] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i !=j), name="BatchTiming1")
model.addConstrs((s_b_var[j] <= c_b_var[i] + M*(1 - s[i,j,v]) for v in V for i in B for j in B if i !=j), name="BatchTiming2")
model.addConstrs((s_b_var[i] <= M * (1 - s[0,i,v]) for v in V for i in B), name="BatchStartFrom0")
model.addConstrs((0 <= s_b_var[i] + M * (1 - s[0,i,v]) for v in V for i in B), name="BatchStartNonNeg")

# Completion times of orders
# If i precedes j in batch b
model.addConstrs((c[j] >= c[i] + DT[i,j] - M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeForward")
model.addConstrs((c[j] <= c[i] + DT[i,j] + M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeBackward")

# First visited order in batch
model.addConstrs((c[i] >= s_b_var[b] + DT[0,i] - M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime1")
model.addConstrs((c[i] <= s_b_var[b] + DT[0,i] + M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime2")

# Last visited order in batch
model.addConstrs((c_b_var[b] >= c[i] + DT[i,0] - M * (1 - r_b[i,0,b]) for b in B for i in I), name="LastOrderTime1")
model.addConstrs((c[i] <= c_b_var[b] + M * (1 - r_b[i,0,b]) - DT[i,0] for b in B for i in I), name="LastOrderTime2")

# Earliness and Tardiness
model.addConstrs((E[i] >= D[i] - c[i] - M * (1 - a[i]) for i in I), name="EarlinessDef")
model.addConstrs((T[i] >= c[i] - D[i] - M * (1 - a[i]) for i in I), name="TardinessDef")

# Disposal
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
