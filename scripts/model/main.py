from gurobipy import Model, GRB, quicksum

# Create the model
model = Model("PharmaceuticalDelivery")

# Sets
I = [1, 2, 3]  # Three pharmaceutical orders
B = [1, 2, 3, 4]   # Two delivery batches
V = [1, 2]     # Two vehicles
N = [0, 1, 2, 3]  # Nodes: depot (0) and three orders (1, 2, 3)
 # Revenue of pharmaceutical orders

# Parameters
R = {1: 100, 2: 150, 3: 200}  # Revenue of pharmaceutical orders
W = {1: 10, 2: 15, 3: 20}  # Weight of pharmaceutical orders in kg
t_lb = {1: 2, 2: 2, 3: 2}  # Lower temperature bound (°C)
t_ub = {1: 8, 2: 8, 3: 8}  # Upper temperature bound (°C)
CA = 30  # Vehicle capacity in kg
ST_LB = {1: 8, 2: 9, 3: 7}  # Lower service time (hours)
ST_UB = {1: 17, 2: 17, 3: 17}  # Upper service time (hours)
M = 1e6  # A large constant

# Decision variables
a = model.addVars(I, vtype=GRB.BINARY, name="a")
x = model.addVars(I, B, vtype=GRB.BINARY, name="x")
y = model.addVars(B, V, vtype=GRB.BINARY, name="y")
z = model.addVars(I, vtype=GRB.BINARY, name="z")
t_b = model.addVars(B, vtype=GRB.CONTINUOUS, lb=t_lb, ub=t_ub, name="t_b")
r_b = model.addVars(N, N, B, vtype=GRB.BINARY, name="r_b")
u = model.addVars(N, B, vtype=GRB.INTEGER, name="u")
s = model.addVars(B, B, V, vtype=GRB.BINARY, name="s")
s_b = model.addVars(B, vtype=GRB.CONTINUOUS, name="s_b")
c_b = model.addVars(B, vtype=GRB.CONTINUOUS, name="c_b")
u_bv = model.addVars(B, V, vtype=GRB.INTEGER, name="u_bv")
E = model.addVars(I, vtype=GRB.CONTINUOUS, name="E")
T = model.addVars(I, vtype=GRB.CONTINUOUS, name="T")

# Define the constant alpha (for example, set alpha = 1 for simplicity)
alpha = 1  # You can adjust this value based on the problem context

# Objective function components
DeliveryRevenue = quicksum(R[i] * a[i] for i in I)  # Delivery revenue
EarlinessPenalty = quicksum(alpha * R[i] * (E[i] + T[i]) for i in I)  # Earliness and tardiness cost
DisposalPenalty = quicksum(R[i] * z[i] for i in I)  # Disposal cost

# Maximize total profit (DeliveryRevenue - EarlinessPenalty - DisposalPenalty)
model.setObjective(DeliveryRevenue - (EarlinessPenalty + DisposalPenalty), GRB.MAXIMIZE)


# Constraints

# Constraint (9)
model.addConstrs((quicksum(x[i, b] for b in B) == a[i] for i in I), name="C9")

# Constraint (10)
model.addConstrs((quicksum(x[i, b] * W[i] for i in I) <= CA for b in B), name="C10")

# Constraint (11)
model.addConstrs((ST_LB[i] - M * (1 - x[i, b]) <= t_b[b] for b in B for i in I), name="C11a")
model.addConstrs((t_b[b] <= ST_UB[i] + M * (1 - x[i, b]) for b in B for i in I), name="C11b")

# Constraint (12)
model.addConstrs((quicksum(y[b, v] for v in V) <= quicksum(x[i, b] for i in I) for b in B), name="C12")

# Constraint (13)
model.addConstrs((x[i, b] <= quicksum(y[b, v] for v in V) for b in B for i in I), name="C13a")
model.addConstrs((1 >= quicksum(y[b,v] for v in V) for b in B), name="C13b")

# Constraint (14)
model.addConstrs((r_b[0, i, b] + quicksum(r_b[i, j, b] for j in I if i != j) == x[i, b] for b in B for i in I), name="C14")

# Constraint (15)
model.addConstrs((r_b[i, 0, b] + quicksum(r_b[i, j, b] for j in I if i != j) == x[i, b] for b in B for i in I), name="C15")

# Constraint (16)
model.addConstrs((quicksum(x[i, b] for i in I) <= M * quicksum(r_b[0, j, b] for j in I) for b in B), name="C16")

# Constraint (17)
model.addConstrs((quicksum(x[i, b] for i in I) <= M * quicksum(r_b[j, 0, b] for j in I) for b in B), name="C17")

# Constraint (18)
model.addConstrs((r_b[i, i, b] == 0 for b in B for i in N), name="C18")

# Constraint (19)
model.addConstrs((u[0, b] == 0 for b in B), name="C19")

# Constraint (20)
model.addConstrs((u[i, b] + r_b[i, j, b] <= u[j, b] + M * (1 - r_b[i, j, b]) for b in B for i in N for j in I if i != j), name="C20")

# Constraint (21)
model.addConstrs((u[i, b] <= M * quicksum(r_b[i, j, b] for j in I if j != i) for b in B for i in I), name="C21")

# Constraint (22)
model.addConstrs((u[i, b] <= x[i, b] for b in B for i in I), name="C22")

# Constraint (23)
model.addConstrs((s[i, j, v] == y[i, v] for v in V for i in B for j in B if i != j), name="C23")

# # Constraint (24)
# model.addConstrs((s[0, j, v] == quicksum(s[i, j, v] for i in B if i != j) for v in V for j in B), name="C24")
#
# # Constraint (25)
# model.addConstrs((s[i, 0, v] == quicksum(s[i, j, v] for j in B if i != j) for v in V for i in B), name="C25")
#
# # Constraint (26)
# model.addConstrs((quicksum(s[i, j, v] for j in B) <= M * quicksum(s[0, j, v] for j in B) for v in V for i in B), name="C26")
#
# # Constraint (27)
# model.addConstrs((s[i, j, v] == 0 for v in V for i in B for j in B if i == j), name="C27")
#
# # Constraint (28)
# model.addConstrs((u_bv[b, v] == 0 for b in B for v in V), name="C28")
#
# # Constraint (29)
# model.addConstrs((u_bv[b, v] <= quicksum(y[b, v] for v in V) for b in B for v in V), name="C29")
#
# # Constraint (30)
# model.addConstrs((c_b[b] <= quicksum(s_b[b] for b in B) for v in V for b in B), name="C30")
#
# # Constraint (31)
# model.addConstrs((s_b[b] <= quicksum(y[b, v] for v in V) for b in B for v in V), name="C31")
#
# # Constraint (32)
# model.addConstrs((c_b[b] + M * (1 - s[i, j, v]) >= s_b[b] for v in V for i in B for j in B if i != j), name="C32")
#
# # Constraint (33)
# model.addConstrs((s_b[b] + M * (1 - s[i, j, v]) >= c_b[b] for v in V for i in B for j in B if i != j), name="C33")
#
# # Constraint (34)
model.addConstrs((s_b[b] <= M * quicksum(y[b, v] for v in V) for b in B), name="C34")
#
# # Constraint (35)
model.addConstrs((c_b[b] <= M * quicksum(y[b, v] for v in V) for b in B), name="C35")
#
# # Constraint (36)
# model.addConstrs((c_b[b] + DT[i] <= s_b[b] + M * (1 - r_b[i, j, b]) for b in B for i in I for j in I if i != j), name="C36")
#
# # Constraint (37)
# model.addConstrs((s_b[b] + DT[j] <= c_b[b] + M * (1 - r_b[i, j, b]) for b in B for i in I for j in I if i != j), name="C37")
#
# # Constraint (38)
# model.addConstrs((c_b[b] + DT[i] <= s_b[b] + M * (1 - r_b[i, j, b]) for b in B for i in I), name="C38")
#
# # Constraint (39)
# model.addConstrs((s_b[b] + DT[i] <= c_b[b] + M * (1 - r_b[i, j, b]) for b in B for i in I), name="C39")
#
# # Constraint (40)
# model.addConstrs((c_b[b] + DT[i] <= s_b[b] + M * (1 - r_b[i, j, b]) for b in B for i in I), name="C40")
#
# # Constraint (41)
# model.addConstrs((s_b[b] + DT[i] <= c_b[b] + M * (1 - r_b[i, j, b]) for b in B for i in I), name="C41")
#
# # Constraint (42)
# model.addConstrs((a[i] + x[i, b] + y[b, v] + z[i, j] <= 1 for i in I for b in B for v in V), name="C42")
#
# # Constraint (43)
model.addConstrs((r_b[i, j, b] >= 0 for i in N for j in N for b in B), name="C43")
#
# # Constraint (44)
model.addConstrs((a[i] >= 0 for i in I), name="C44")
#
# # Constraint (45)
model.addConstrs((x[i, b] >= 0 for i in I for b in B), name="C45")
#
# # Constraint (46)
model.addConstrs((y[b, v] >= 0 for b in B for v in V), name="C46")
#
# # Constraint (47)
model.addConstrs((u_bv[b, v] >= 0 for b in B for v in V), name="C47")

# Optimize the model
model.optimize()

# Print the solution
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    for var in model.getVars():
        if var.x > 0:
            print(f"{var.varName}: {var.x}")
