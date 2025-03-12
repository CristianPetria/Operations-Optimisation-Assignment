from gurobipy import Model, GRB, quicksum
import random
import time


def generate_random_travel_times(num_nodes, max_time=20):
    DT = {}
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if i == j:
                DT[(i, j)] = 0
            else:
                time = random.randint(1, max_time)
                DT[(i, j)] = time
                DT[(j, i)] = time
    return DT

def generate_random_dict(num_entries, value_range):
    return {i: random.randint(value_range[0], value_range[1]) for i in range(1, num_entries + 1)}

def generate_list(start, end):
    return list(range(start, end + 1))

def create_base_settings(num_orders=20):
    random.seed(1)
    I = generate_list(1, num_orders)
    B = generate_list(1, 5)
    B0 = [0] + B
    V = generate_list(1, 8)
    N = [0] + I
    R = generate_random_dict(len(I), (1000, 2000)) 
    W = generate_random_dict(len(I), (5, 25))
    t_lb = generate_random_dict(len(I), (1, 5))
    t_ub = generate_random_dict(len(I), (6, 15))
    ST_LB = generate_random_dict(len(I), (6, 10))
    ST_UB = generate_random_dict(len(I), (15, 20))
    D = generate_random_dict(len(I), (5, 15))
    S = generate_random_dict(len(I), (15, 25))
    DT = generate_random_travel_times(len(N), max_time=20)
    
    return {
        "I": I,
        "B": B,
        "B0": B0,
        "V": V,
        "N": N,
        "R": R,
        "W": W,
        "t_lb": t_lb,
        "t_ub": t_ub,
        "CA": 25,
        "ST_LB": ST_LB,
        "ST_UB": ST_UB,
        "D": D,
        "S": S,
        "alpha": 1,
        "M": 1e7,
        "DT": DT
    }

def create_optimization_model(settings):
    I = settings.get("I")
    B = settings.get("B")
    B0 = settings.get("B0")
    V = settings.get("V")
    N = settings.get("N")
    R = settings.get("R")
    W = settings.get("W")
    t_lb = settings.get("t_lb")
    t_ub = settings.get("t_ub")
    CA = settings.get("CA")
    ST_LB = settings.get("ST_LB")
    ST_UB = settings.get("ST_UB")
    D = settings.get("D")
    S = settings.get("S")
    alpha = settings.get("alpha")
    M = settings.get("M")
    DT = settings.get("DT")
    
    model = Model("PharmaceuticalDelivery")
    
    a = model.addVars(I, vtype=GRB.BINARY, name="a")     # Order acceptance
    x = model.addVars(I, B, vtype=GRB.BINARY, name="x")  # Assign order i to batch b
    y = model.addVars(B, V, vtype=GRB.BINARY, name="y")  # Assign batch b to vehicle v
    z = model.addVars(I, vtype=GRB.BINARY, name="z")     # Disposal indicator
    t_b = model.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="t_b")
    r_b = model.addVars(N, N, B, vtype=GRB.BINARY, name="r_b") 
    u_i = model.addVars(N, B, vtype=GRB.INTEGER, name="u_i")   
    s = model.addVars(B0, B0, V, vtype=GRB.BINARY, name="s")      
    s_b_var = model.addVars(B, vtype=GRB.CONTINUOUS, name="s_b")
    c_b_var = model.addVars(B, vtype=GRB.CONTINUOUS, name="c_b") 
    u_bv = model.addVars(B0, V, vtype=GRB.INTEGER, name="u_bv")   
    E = model.addVars(I, vtype=GRB.CONTINUOUS, name="E")        
    T = model.addVars(I, vtype=GRB.CONTINUOUS, name="T")         
    c = model.addVars(I, vtype=GRB.CONTINUOUS, name="c")        
    
    # Objective function
    DeliveryRevenue = quicksum(R[i]*a[i] for i in I)
    EarlinessTardinessCost = quicksum(alpha*R[i]*(E[i] + T[i]) for i in I)
    DisposalCost = quicksum(R[i]*z[i] for i in I)
    model.setObjective(DeliveryRevenue - (EarlinessTardinessCost + DisposalCost), GRB.MAXIMIZE)

    # Add constraints in exactly the same order as the original code
    
    # Order assignment (9)
    model.addConstrs((quicksum(x[i,b] for b in B) == a[i] for i in I), name="AssignOrderToBatch")

    # Capacity (10)
    model.addConstrs((quicksum(x[i,b]*W[i] for i in I) <= CA for b in B), name="Capacity")

    # Temperature feasibility (11)
    model.addConstrs((t_lb[i] - M*(1 - x[i,b]) <= t_b[b] for i in I for b in B), name="TempLower")
    model.addConstrs((t_b[b] <= t_ub[i] + M*(1 - x[i,b]) for i in I for b in B), name="TempUpper")

    # Batch to vehicle assignment (12 - 13)
    model.addConstrs((quicksum(y[b,v] for v in V) <= quicksum(x[i,b] for i in I) for b in B), name="NonemptyBatchIfAssigned")
    model.addConstrs((x[i,b] <= quicksum(y[b,v] for v in V) for i in I for b in B), name="LinkBatchVehicle")
    model.addConstrs((quicksum(y[b,v] for v in V) <= 1 for b in B), name="OneVehiclePerBatch")

    # Routing constraints (14 - 15)
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
    model.addConstrs((c[j] >= c[i] + DT[i,j] - M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeForward")
    model.addConstrs((c[j] <= c[i] + DT[i,j] + M * (1 - r_b[i,j,b]) for b in B for i in I for j in I if i != j), name="OrderTimeBackward")

    # First visited order in batch (38 - 39)
    model.addConstrs((c[i] >= s_b_var[b] + DT[0,i] - M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime1")
    model.addConstrs((c[i] <= s_b_var[b] + DT[0,i] + M * (1 - r_b[0,i,b]) for b in B for i in I), name="FirstOrderTime2")

    # Last visited order in batch (40 - 41)
    # NOTE: The order of these constraints is matched exactly with original code
    model.addConstrs((c[i] >= c_b_var[b] - M * (1 - r_b[i,0,b]) - DT[i,0] for b in B for i in I), name="LastOrderTime2")
    model.addConstrs((c_b_var[b] >= c[i] + DT[i,0] - M * (1 - r_b[i,0,b]) for b in B for i in I), name="LastOrderTime1")
    
    # Earliness and Tardiness (3 - 4)
    model.addConstrs((E[i] >= D[i] - c[i] - M * (1 - a[i]) for i in I), name="EarlinessDef")
    model.addConstrs((T[i] >= c[i] - D[i] - M * (1 - a[i]) for i in I), name="TardinessDef")
    
    # Add BOTH disposal constraints from main_attempt1.py
    # 1. Disposal if delivered after shelf life
    model.addConstrs((c[i] - S[i] <= M * z[i] for i in I), name="DisposalDef")
    
    # 2. Link disposal to order rejection
    model.addConstrs((z[i] >= 1 - a[i] for i in I), name="LinkDisposalLower")
    model.addConstrs((z[i] <= 1 - a[i] for i in I), name="LinkDisposalUpper")
    variables = {
        "a": a, "x": x, "y": y, "z": z, "t_b": t_b, "r_b": r_b,
        "u_i": u_i, "s": s, "s_b_var": s_b_var, "c_b_var": c_b_var,
        "u_bv": u_bv, "E": E, "T": T, "c": c
    }
    
    return model, variables

def run_model(settings, time_limit=3000, presolve=0, verbose=True):
    model, variables = create_optimization_model(settings)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.Presolve, presolve)
    model.setParam(GRB.Param.OutputFlag, 1 if verbose else 0)
    model.setParam(GRB.Param.LogFile, "gurobi_log.txt")
    
    a = variables["a"]
    I = settings.get("I") 
    def a_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            print(f"\nNew feasible solution found with objective value: {obj_val}")
            a_vars = [variables["a"][i] for i in I]
            a_values = model.cbGetSolution(a_vars)
            accepted_orders = [i for i, val in zip(I, a_values) if val > 0.5]
            rejected_orders = [i for i, val in zip(I, a_values) if val <= 0.5]
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    results = {
        'objective': float('nan'),
        'accepted': 0,
        'rejected': 0,
        'solve_time': solve_time,
        'gap': float('inf'),
        'status': model.status
    }
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            results['objective'] = model.objVal
            results['gap'] = model.MIPGap
            for i in I:
                if a[i].X > 0.5:
                    results['accepted'] += 1
                else:
                    results['rejected'] += 1

    return results