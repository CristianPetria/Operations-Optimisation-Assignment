import unittest
from gurobipy import Model, GRB, quicksum
import sys
import os

# Go up two levels from tests folder to scripts folder
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from model.main_attempt1 import generate_random_travel_times, I, B, B0, V, N, R, W, t_lb, t_ub, CA, ST_LB, ST_UB, D, S, alpha, M, DT

class TestPharmaceuticalDeliveryModel(unittest.TestCase):



    def setUp(self):
        self.model = Model("TestModel")

    # Test if the data is initialized correctly
    def test_data_initialization(self):
        self.assertEqual(len(I), 20)
        self.assertEqual(len(B), 5)
        self.assertEqual(len(B0), 6)
        self.assertEqual(len(V), 8)
        self.assertEqual(len(N), 21)

        self.assertEqual(CA, 25)
        self.assertEqual(alpha, 1)
        self.assertEqual(M, 1e7)

        for data in [R, W, t_lb, t_ub, ST_LB, ST_UB, D, S]:
            self.assertEqual(len(data), 20)

        self.assertTrue(all(0 <= DT[i, j] <= 20 for i in N for j in N if i != j))
        self.assertTrue(all(DT[i, j] == DT[j, i] for i in N for j in N if i != j))

    def test_order_acceptance_variable(self):
        a = self.model.addVars(I, vtype=GRB.BINARY, name="a")
        self.assertEqual(len(a), len(I))

    def test_batch_assignment_variable(self):
        x = self.model.addVars(I, B, vtype=GRB.BINARY, name="x")
        self.assertEqual(len(x), len(I) * len(B))

    def test_vehicle_assignment_variable(self):
        y = self.model.addVars(B, V, vtype=GRB.BINARY, name="y")
        self.assertEqual(len(y), len(B) * len(V))

    def test_disposal_indicator_variable(self):
        z = self.model.addVars(I, vtype=GRB.BINARY, name="z")
        self.assertEqual(len(z), len(I))

    def test_batch_temperature_variable(self):
        t_b = self.model.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="t_b")
        self.assertEqual(len(t_b), len(B))

    def test_routing_within_batch_variable(self):
        r_b = self.model.addVars(N, N, B, vtype=GRB.BINARY, name="r_b")
        self.assertEqual(len(r_b), len(N) * len(N) * len(B))

    def test_subtour_elimination_routing_variable(self):
        u_i = self.model.addVars(N, B, vtype=GRB.INTEGER, name="u_i")
        self.assertEqual(len(u_i), len(N) * len(B))

    def test_batch_scheduling_variable(self):
        s = self.model.addVars(B0, B0, V, vtype=GRB.BINARY, name="s")
        self.assertEqual(len(s), len(B0) * len(B0) * len(V))

    def test_batch_start_time_variable(self):
        s_b_var = self.model.addVars(B, vtype=GRB.CONTINUOUS, name="s_b")
        self.assertEqual(len(s_b_var), len(B))

    def test_batch_completion_time_variable(self):
        c_b_var = self.model.addVars(B, vtype=GRB.CONTINUOUS, name="c_b")
        self.assertEqual(len(c_b_var), len(B))

    def test_subtour_elimination_scheduling_variable(self):
        u_bv = self.model.addVars(B0, V, vtype=GRB.INTEGER, name="u_bv")
        self.assertEqual(len(u_bv), len(B0) * len(V))

    def test_earliness_variable(self):
        E = self.model.addVars(I, vtype=GRB.CONTINUOUS, name="E")
        self.assertEqual(len(E), len(I))

    def test_tardiness_variable(self):
        T = self.model.addVars(I, vtype=GRB.CONTINUOUS, name="T")
        self.assertEqual(len(T), len(I))

    def test_completion_time_variable(self):
        c = self.model.addVars(I, vtype=GRB.CONTINUOUS, name="c")
        self.assertEqual(len(c), len(I))

    def test_travel_time_matrix(self):
        self.assertEqual(len(DT), len(N) ** 2)

    def test_objective_function(self):
        # Random sample values for testing
        R = {0: 150, 1: 250}
        I = [0, 1]
        alpha = 0.1

        # Add decision variables to the model
        a = self.model.addVars(I, vtype=GRB.BINARY, name="a")
        E = self.model.addVars(I, vtype=GRB.CONTINUOUS, name="E")
        T = self.model.addVars(I, vtype=GRB.CONTINUOUS, name="T")
        z = self.model.addVars(I, vtype=GRB.BINARY, name="z")

        self.model.update()

        DeliveryRevenue = quicksum(R[i] * a[i] for i in I)
        EarlinessTardinessCost = quicksum(alpha * R[i] * (E[i] + T[i]) for i in I)
        DisposalCost = quicksum(R[i] * z[i] for i in I)
        objective = DeliveryRevenue - (EarlinessTardinessCost + DisposalCost)
        self.model.setObjective(objective, GRB.MAXIMIZE)

        sample_values = {a[0]: 1, a[1]: 1, E[0]: 1, E[1]: 3, T[0]: 2, T[1]: 0, z[0]: 0, z[1]: 1}

        for var, value in sample_values.items():
            var.setAttr("Start", value)

        # To avoid Gurobi tweaking values
        self.model.setParam(GRB.Param.Presolve, 0)  # To avoid Gurobi tweaking values
        self.model.update()

        # Fix the variables to the sample values
        for var, value in sample_values.items():
            var.setAttr(GRB.Attr.LB, value)
            var.setAttr(GRB.Attr.UB, value)

        self.model.optimize()

        # Expected value of the objective function
        expected_value = (R[0] * sample_values[a[0]] + R[1] * sample_values[a[1]]) - (
                alpha * R[0] * (sample_values[E[0]] + sample_values[T[0]]) +
                alpha * R[1] * (sample_values[E[1]] + sample_values[T[1]]) +
                R[1] * sample_values[z[1]]
        )

        # Check if the objective value is as expected
        self.assertAlmostEqual(self.model.getObjective().getValue(), expected_value)



# Constraint (9 - 11)
class TestPharmaceuticalDeliveryConstraints(unittest.TestCase):

    def setUp(self):
        self.model = Model("TestModel")

        # Dummy data
        self.I = [1, 2, 3]  # Orders
        self.B = [1]  # Batches
        self.W = {1: 5, 2: 10, 3: 4}  # Weights
        self.CA = 15  # Capacity
        self.t_lb = {1: 2, 2: 4, 3: 3}  # Lower temp bounds
        self.t_ub = {1: 8, 2: 10, 3: 7}  # Upper temp bounds
        self.M = 1000  # Large constant

        # Decision variables
        self.a = self.model.addVars(self.I, vtype=GRB.BINARY, name="a")  # Order acceptance
        self.x = self.model.addVars(self.I, self.B, vtype=GRB.BINARY, name="x")  # Batch assignment
        self.t_b = self.model.addVars(self.B, vtype=GRB.CONTINUOUS, name="t_b")  # Batch temperature

        # Constraints
        # Order Assignment
        self.model.addConstrs(
            (quicksum(self.x[i, b] for b in self.B) == self.a[i] for i in self.I),
            name="AssignOrderToBatch"
        )

        # Capacity Constraint
        self.model.addConstrs(
            (quicksum(self.x[i, b] * self.W[i] for i in self.I) <= self.CA for b in self.B),
            name="Capacity"
        )

        # Temperature Feasibility
        self.model.addConstrs(
            (self.t_lb[i] - self.M * (1 - self.x[i, b]) <= self.t_b[b] for i in self.I for b in self.B),
            name="TempLower"
        )
        self.model.addConstrs(
            (self.t_b[b] <= self.t_ub[i] + self.M * (1 - self.x[i, b]) for i in self.I for b in self.B),
            name="TempUpper"
        )

        self.model.update()

    def test_order_assignment(self):
        """ Test that each order is assigned to exactly one batch if it's accepted """
        self.model.optimize()
        for i in self.I:
            assigned_batches = sum(self.x[i, b].X for b in self.B)
            self.assertAlmostEqual(assigned_batches, self.a[i].X)

    def test_capacity_constraint(self):
        """ Test that no batch exceeds the weight capacity """
        self.model.optimize()
        for b in self.B:
            total_weight = sum(self.x[i, b].X * self.W[i] for i in self.I)
            self.assertLessEqual(total_weight, self.CA)

    def test_temperature_feasibility(self):
        """ Test that assigned orders respect their temperature constraints """
        self.model.optimize()
        for i in self.I:
            for b in self.B:
                if self.x[i, b].X > 0.5:  # If order i is assigned to batch b
                    self.assertGreaterEqual(self.t_b[b].X, self.t_lb[i])
                    self.assertLessEqual(self.t_b[b].X, self.t_ub[i])
# Constraint (12 - 13)
class TestBatchVehicleAssignment(unittest.TestCase):

    def setUp(self):
        self.model = Model("BatchVehicleAssignment")

        # Dummy data for testing
        self.I = [1, 2, 3]  # Orders
        self.B = [1, 2]  # Batches
        self.V = [1, 2]  # Vehicles

        # Decision variables
        self.x = self.model.addVars(self.I, self.B, vtype=GRB.BINARY, name="x")  # Order to batch
        self.y = self.model.addVars(self.B, self.V, vtype=GRB.BINARY, name="y")  # Batch to vehicle

        # Constraints
        self.model.addConstrs(
            (quicksum(self.y[b, v] for v in self.V) <= quicksum(self.x[i, b] for i in self.I) for b in self.B),
            name="NonemptyBatchIfAssigned")
        self.model.addConstrs(
            (self.x[i, b] <= quicksum(self.y[b, v] for v in self.V) for i in self.I for b in self.B),
            name="LinkBatchVehicle")
        self.model.addConstrs((quicksum(self.y[b, v] for v in self.V) <= 1 for b in self.B),
                              name="OneVehiclePerBatch")

        self.model.update()

    def test_nonempty_batch_if_assigned(self):
        """Test that a batch is assigned to a vehicle only if it contains at least one order."""
        self.model.optimize()
        for b in self.B:
            assigned_vehicle = sum(self.y[b, v].X for v in self.V)
            assigned_orders = sum(self.x[i, b].X for i in self.I)
            self.assertGreaterEqual(assigned_orders, assigned_vehicle)

    def test_link_batch_vehicle(self):
        """Test that an order is only assigned to a batch if that batch is assigned to a vehicle."""
        self.model.optimize()
        for i in self.I:
            for b in self.B:
                batch_assigned = sum(self.y[b, v].X for v in self.V)
                self.assertGreaterEqual(batch_assigned, self.x[i, b].X)

    def test_one_vehicle_per_batch(self):
        """Test that a batch is assigned to at most one vehicle."""
        self.model.optimize()
        for b in self.B:
            total_vehicles = sum(self.y[b, v].X for v in self.V)
            self.assertLessEqual(total_vehicles, 1)

class TestBatchVehicleAssignment(unittest.TestCase):

    def setUp(self):
        self.model = Model("BatchVehicleAssignment")

        # Dummy data for testing
        self.I = [1, 2, 3]  # Orders
        self.B = [1, 2]  # Batches
        self.V = [1, 2]  # Vehicles
        self.N = [0] + self.I  # Including depot (0)
        self.M = 1000  # Large constant for subtour elimination

        # Decision variables
        self.x = self.model.addVars(self.I, self.B, vtype=GRB.BINARY, name="x")  # Order to batch
        self.y = self.model.addVars(self.B, self.V, vtype=GRB.BINARY, name="y")  # Batch to vehicle
        self.r_b = self.model.addVars(self.N, self.N, self.B, vtype=GRB.BINARY, name="r_b")  # Routing
        self.u_i = self.model.addVars(self.I, self.B, vtype=GRB.CONTINUOUS, name="u_i")  # Order position in route

        # Constraints
        self.model.addConstrs(
            (quicksum(self.y[b, v] for v in self.V) <= quicksum(self.x[i, b] for i in self.I) for b in self.B),
            name="NonemptyBatchIfAssigned")
        self.model.addConstrs(
            (self.x[i, b] <= quicksum(self.y[b, v] for v in self.V) for i in self.I for b in self.B),
            name="LinkBatchVehicle")
        self.model.addConstrs((quicksum(self.y[b, v] for v in self.V) <= 1 for b in self.B),
                              name="OneVehiclePerBatch")

        # Routing constraints
        self.model.addConstrs(
            (self.r_b[0, i, b] + quicksum(self.r_b[j, i, b] for j in self.I if j != i) == self.x[i, b] for b in
             self.B for i in self.I), name="RoutingInflow")
        self.model.addConstrs(
            (self.r_b[i, 0, b] + quicksum(self.r_b[i, j, b] for j in self.I if j != i) == self.x[i, b] for b in
             self.B for i in self.I), name="RoutingOutflow")
        self.model.addConstrs(
            (quicksum(self.x[i, b] for i in self.I) <= self.M * quicksum(self.r_b[0, j, b] for j in self.I) for b in
             self.B), name="StartFromDepot")
        self.model.addConstrs(
            (quicksum(self.x[i, b] for i in self.I) <= self.M * quicksum(self.r_b[j, 0, b] for j in self.I) for b in
             self.B), name="ReturnToDepot")
        self.model.addConstrs((self.r_b[i, i, b] == 0 for b in self.B for i in self.N), name="NoSelfLoop")
        self.model.addConstrs((self.u_i[0, b] == 0 for b in self.B), name="DepotOrderPosition")
        self.model.addConstrs(
            (self.u_i[i, b] + self.r_b[i, j, b] <= self.u_i[j, b] + self.M * (1 - self.r_b[i, j, b]) for b in self.B
             for i in self.N for j in self.I if i != j), name="RoutingSubtour1")
        self.model.addConstrs(
            (self.u_i[i, b] <= self.M * quicksum(self.r_b[i, j, b] for j in self.I if j != i) for b in self.B for i
             in self.I), name="RoutingSubtour2")
        self.model.addConstrs(
            (self.u_i[i, b] <= quicksum(self.x[k, b] for k in self.I) for b in self.B for i in self.I),
            name="RoutingSubtour3")

        self.model.update()

    def test_nonempty_batch_if_assigned(self):
        """Test that a batch is assigned to a vehicle only if it contains at least one order."""
        self.model.optimize()
        for b in self.B:
            assigned_vehicle = sum(self.y[b, v].X for v in self.V)
            assigned_orders = sum(self.x[i, b].X for i in self.I)
            self.assertGreaterEqual(assigned_orders, assigned_vehicle)

    def test_link_batch_vehicle(self):
        """Test that an order is only assigned to a batch if that batch is assigned to a vehicle."""
        self.model.optimize()
        for i in self.I:
            for b in self.B:
                batch_assigned = sum(self.y[b, v].X for v in self.V)
                self.assertGreaterEqual(batch_assigned, self.x[i, b].X)

    def test_one_vehicle_per_batch(self):
        """Test that a batch is assigned to at most one vehicle."""
        self.model.optimize()
        for b in self.B:
            total_vehicles = sum(self.y[b, v].X for v in self.V)
            self.assertLessEqual(total_vehicles, 1)

    def test_no_self_loop(self):
        """Test that there are no self-loops in routing."""
        self.model.optimize()
        for b in self.B:
            for i in self.N:
                self.assertEqual(self.r_b[i, i, b].X, 0)

    def test_start_and_end_at_depot(self):
        """Test that a nonempty batch starts and ends at the depot."""
        self.model.optimize()
        for b in self.B:
            assigned_orders = sum(self.x[i, b].X for i in self.I)
            if assigned_orders > 0:
                self.assertGreater(sum(self.r_b[0, j, b].X for j in self.I), 0)
                self.assertGreater(sum(self.r_b[j, 0, b].X for j in self.I), 0)

if __name__ == "__main__":
    unittest.main()
