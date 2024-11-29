from gurobipy import Model, GRB

# Define sets
I = range(5)  # Orders
B = range(3)  # Batches
V = range(2)  # Vehicles
D = range(3)
J = range(5)

# Define parameters
R = {i: 100 for i in I}
W = {i: 10 for i in I}
CA = {b: 50 for b in B}
DTij = {(i, j): 10 for i in I for j in I}

# Create a model
model = Model()

# Add variables
x = model.addVars(I, B, vtype=GRB.BINARY, name="x")
y = model.addVars(B, V, vtype=GRB.BINARY, name="y")
z = model.addVars(I, vtype=GRB.BINARY, name="z")


# Add continuous variables
t = model.addVars(B, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="t")  # Example: Temperature
c = model.addVars(I, vtype=GRB.CONTINUOUS, name="c")                # Example: Completion time
E = model.addVars(I, vtype=GRB.CONTINUOUS, name="E")                # Example: Earliness
T = model.addVars(I, vtype=GRB.CONTINUOUS, name="T")                # Example: Tardiness

# Define variables (add these to your model setup)
a = model.addVars(I, vtype=GRB.BINARY, name="a")  # Binary: order assigned
r = model.addVars(B, I, I, vtype=GRB.BINARY, name="r")  # Routing variables
u = model.addVars(B, I, vtype=GRB.CONTINUOUS, name="u")  # Sequence variables
st = model.addVars(B, lb=0, ub=100, vtype=GRB.CONTINUOUS, name="st")  # Start times
sb = model.addVars(V, I, lb=0, vtype=GRB.CONTINUOUS, name="sb")  # Sub-tour variables
ci = model.addVars(B, I, vtype=GRB.CONTINUOUS, name="ci")  # Completion time start
cj = model.addVars(B, I, vtype=GRB.CONTINUOUS, name="cj")  # Completion time end
cb = model.addVars(B, vtype=GRB.CONTINUOUS, name="cb")  # Batch completion time
s = model.addVars(V, I, I, vtype=GRB.BINARY, name="s")

# Define ST_lb (lower bound for start times)
ST_lb = {b: 0 for b in B}  # Example: all batches start at time 0 initially

# Modify Constraint (9) to align with 'a' variable
for i in I:
    model.addConstr(sum(x[i, b] for b in B) == a[i], name=f"Constraint_9")

# Example parameters
R = {i: 100 for i in I}  # Revenue for orders
M = 1000  # Disposal cost penalty

# Define ETC and DFC (sub-components of the objective)
ETC = sum(R[i] * (E[i] + T[i]) for i in I)
DFC = sum(M * z[i] for i in I)

# Objective function
model.setObjective(sum(R[i] * x[i, b] for i in I for b in B) - ETC - DFC, GRB.MAXIMIZE)

# Constraint (9): Ensure each order is assigned to at most one batch
for i in I:
    model.addConstr(sum(x[i, b] for b in B) == a[i], name=f"Constraint_9")

# Constraint (10): Batch capacity constraint
for b in B:
    model.addConstr(sum(W[i] * x[i, b] for i in I) <= CA[b], name=f"Constraint_10")

# Constraint (11): Starting time constraint for orders in batches
for b in B:
    for i in I:
        model.addConstr(
            st[b] >= M * x[i, b] - ST_lb[b] + M * (1 - a[i]), name=f"Constraint_11"
        )

# Constraint (12): Ensure each batch is assigned to exactly one vehicle
for b in B:
    model.addConstr(sum(y[b, v] for v in V) == 1, name=f"Constraint_12")

# Constraint (13): Ensure z[i] is set to 0 if an order is delivered
for i in I:
    model.addConstr(z[i] <= sum(x[i, b] for b in B), name=f"Constraint_13")

# Constraint (14): Routing constraints (precedence for nodes)
for b in B:
    for i in I:
        for j in I:
            if i != j:
                model.addConstr(r[b, i, j] + r[b, j, i] <= 1, name=f"Constraint_14")

# Constraint (15): Ensure node precedence for orders in a batch
for b in B:
    for i in I:
        model.addConstr(
            sum(r[b, i, j] for j in I if i != j)
            == sum(r[b, j, i] for j in I if i != j),
            name=f"Constraint_15",
        )

# Constraint (16): Ensure a node appears in the routing only once
for b in B:
    model.addConstr(
        sum(r[b, i, j] for i in I for j in I if i != j) <= len(I),
        name=f"Constraint_16",
    )

# Constraint (17): Ensure routing precedence does not conflict with batch assignment
for b in B:
    for i in I:
        model.addConstr(
            sum(r[b, i, j] for j in I if i != j) <= M * x[i, b], name=f"Constraint_17"
        )

# Constraint (18): Precedence constraints for routes
for b in B:
    model.addConstr(r[b, 0, 0] == 0, name=f"Constraint_18")

# Constraint (19): Sequence variable initialization
for b in B:
    for i in I:
        model.addConstr(u[b, i] == 0, name=f"Constraint_19")

# Constraint (20): Enforce sequence precedence for routing
for b in B:
    for i in I:
        for j in I:
            if i != j:
                model.addConstr(
                    u[b, j] >= u[b, i] + 1 - M * (1 - r[b, i, j]),
                    name=f"Constraint_20",
                )

# Constraint (21): Sequence bounds for routing variables
for b in B:
    for i in I:
        model.addConstr(u[b, i] <= M * sum(r[b, i, j] for j in I if i != j), name=f"Constraint_21")

# Constraint (22): Vehicle sequence initialization
for b in B:
    model.addConstr(sum(y[b, v] for v in V) <= len(V), name=f"Constraint_22")

# Constraint (23): Vehicle routing constraints
for b in B:
    for v in V:
        for i in I:
            for j in I:
                if i != j:
                    model.addConstr(
                        s[v, i, j] <= r[b, i, j] + M * (1 - y[b, v]),
                        name=f"Constraint_23",
                    )

# Constraint (24): Precedence constraints between batches
for b in B:
    model.addConstr(sum(s[v, i, j] for v in V) <= len(V), name=f"Constraint_24")

# Constraint (25): Precedence within vehicles for routes
for b in B:
    for v in V:
        for i in I:
            model.addConstr(sum(s[v, i, j] for j in I if i != j) <= len(V), name=f"Constraint_25")

# Constraint (26): Precedence conflict avoidance between vehicles
for b in B:
    for v in V:
        for i in I:
            for j in I:
                if i != j:
                    model.addConstr(
                        s[v, i, j] <= 1 - y[b, v], name=f"Constraint_26"
                    )

# Constraint (27): Ensure vehicle precedence initialization
for v in V:
    model.addConstr(u[v, 0] == 0, name=f"Constraint_27")

# Constraint (28): Scheduling sequence variable bounds
for b in B:
    for v in V:
        for i in I:
            for j in I:
                if i != j:
                    model.addConstr(
                        u[v, j] >= u[v, i] + 1 - M * (1 - s[v, i, j]),
                        name=f"Constraint_28",
                    )

# Constraint (29): Ensure routing sequence within batches for vehicles
for b in B:
    for i in I:
        for j in I:
            if i != j:
                model.addConstr(
                    u[v, j] <= M * sum(r[b, i, j] for v in V), name=f"Constraint_29"
                )

# Constraint (30): Vehicle sequence bounds for scheduling
for b in B:
    model.addConstr(
        u[v, b] <= M * sum(s[v, i, j] for i in I for j in I if i != j),
        name=f"Constraint_30",
    )

# Constraint (31): Vehicle assignment constraint for routing
for v in V:
    for i in list(I) + list(B) + list(D) + list(E):
        model.addConstr(u[v, i] <= sum(y[b, v] for b in B), name=f"Constraint_31")

# Constraint (32): Sub-tour elimination (part 1)
for v in V:
    for i in I:
        for j in B:
            if i != j:
                model.addConstr(sb[v, i] <= sb[v, j] + M * (1 - s[v, i, j]), name=f"Constraint_32")

# Constraint (33): Sub-tour elimination (part 2)
for v in V:
    for i in I:
        for j in B:
            if i != j:
                model.addConstr(sb[v, j] <= sb[v, i] + M * (1 - s[v, i, j]), name=f"Constraint_33")

# Constraint (34): Starting time constraints within batches
for v in V:
    for i in I:
        model.addConstr(sb[v, i] <= sb[v, i] + M * (1 - s[v, i, j]), name=f"Constraint_34")

# Constraint (35): Boundaries for the sub-tour variables
for v in V:
    for i in I:
        model.addConstr(0 <= sb[v, i] + M * (1 - s[v, i, j]), name=f"Constraint_35")

# Constraint (36): Arrival time constraints (part 1)
for b in B:
    for i in I:
        for j in I:
            if i != j:
                model.addConstr(
                    ci[b, i] + DTij[i, j] <= cj[b, j] + M * (1 - r[b, i, j]),
                    name=f"Constraint_36",
                )

# Constraint (37): Arrival time constraints (part 2)
for b in B:
    for i in I:
        for j in I:
            if i != j:
                model.addConstr(
                    cj[b, j] >= ci[b, i] + DTij[i, j] - M * (1 - r[b, i, j]),
                    name=f"Constraint_37",
                )

# Constraint (38): Completion time constraints (part 1)
for b in B:
    for i in I:
        model.addConstr(
            ci[b, i] + DTij[i, i] <= ci[b, i] + M * (1 - r[b, i, i]), name=f"Constraint_38"
        )

# Constraint (39): Completion time constraints (part 2)
for b in B:
    for i in I:
        model.addConstr(
            cj[b, i] <= ci[b, i] + DTij[i, i] - M * (1 - r[b, i, i]), name=f"Constraint_39"
        )

# Constraint (40): Completion time of last node in a batch
for b in B:
    for i in I:
        model.addConstr(
            cj[b, i] <= ci[b, i] + DTij[i, i] + c[b] + M * (1 - r[b, i, i]),
            name=f"Constraint_40",
        )

# Constraint (41): Boundaries for the completion time
for b in B:
    for i in I:
        model.addConstr(cj[b, i] <= ci[b, i] + DTij[i, i] + cb[b], name=f"Constraint_41")

# Constraint (42): Binary constraints for decision variables
for i in I:
    for b in B:
        for v in V:
            model.addConstr(a[i] >= 0, name=f"Constraint_42")
            model.addConstr(x[i, b] >= 0, name=f"Constraint_42")
            model.addConstr(y[b, v] >= 0, name=f"Constraint_42")

# Constraint (43): Routing binary variable constraints
for i in list(I) + list(J):
    for j in list(I) + list(J):
        model.addConstr(r[i, j, b] in [0, 1], name=f"Constraint_43")

# Constraint (44): Routing variable bounds for vehicles
for b in B:
    for v in V:
        model.addConstr(r[v, b] in [0, 1], name=f"Constraint_44")

# Constraint (45): Boundaries for the sub-tour variables
for b in B:
    for v in V:
        for i in I + D:
            model.addConstr(sb[v, i] >= 0, name=f"Constraint_45")

# Constraint (46): Routing variable constraints between batches
for b in B:
    for i in I:
        for j in J:
            model.addConstr(s[v, i, j] in [0, 1], name=f"Constraint_46")

# Constraint (47): Non-negativity constraints for vehicle sub-tour variables
for b in B:
    for v in V:
        model.addConstr(u[v, b] >= 0, name=f"Constraint_47")
