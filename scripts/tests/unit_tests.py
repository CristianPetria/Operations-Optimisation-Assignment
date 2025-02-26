import unittest
from gurobipy import Model, GRB, quicksum
import sys
import os

# Go up two levels from tests folder to scripts folder
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from model.main_attempt1 import generate_random_travel_times, I, B, B0, V, N, R, W, t_lb, t_ub, CA, ST_LB, ST_UB, D, S, alpha, M, DT

class TestInitialisation(unittest.TestCase):

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



# Test order assignment constraints (9) and capacity constraint (10)
# and temperature feasibility constraints (11)
class TestCapacityandTemperatureFeasibility(unittest.TestCase):

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

# Routing constraints (14 - 17) test class
class TestRouting(unittest.TestCase):

    def setUp(self):
        self.model = Model("TestRoutingModel")

        # Simplified test data
        self.I = [1, 2, 3]  # Orders
        self.B = [1]  # Single batch for simplicity
        self.N = [0, 1, 2, 3]  # Nodes: 0 = depot, 1-3 = orders

        # Decision variables
        self.x = self.model.addVars(self.I, self.B, vtype=GRB.BINARY, name="x")  # Order to batch
        self.r_b = self.model.addVars(self.N, self.N, self.B, vtype=GRB.BINARY, name="r_b")  # Routing

        # Constraints
        # Flow balance for accepted orders (14-15)
        self.model.addConstrs(
            (self.r_b[0, i, b] + quicksum(self.r_b[j, i, b] for j in self.I if j != i) == self.x[i, b]
             for b in self.B for i in self.I),
            name="RoutingInflow"
        )
        self.model.addConstrs(
            (self.r_b[i, 0, b] + quicksum(self.r_b[i, j, b] for j in self.I if j != i) == self.x[i, b]
             for b in self.B for i in self.I),
            name="RoutingOutflow"
        )

        # Nonempty batch: route must start and end at depot (16-17)
        self.model.addConstrs(
            (quicksum(self.x[i, b] for i in self.I) <= M * quicksum(self.r_b[0, j, b] for j in self.I)
             for b in self.B),
            name="StartFromDepot"
        )
        self.model.addConstrs(
            (quicksum(self.x[i, b] for i in self.I) <= M * quicksum(self.r_b[j, 0, b] for j in self.I)
             for b in self.B),
            name="ReturnToDepot"
        )

        # Objective function (to make model feasible)
        self.model.setObjective(quicksum(self.x[i, b] for i in self.I for b in self.B), GRB.MAXIMIZE)

        self.model.update()

    def test_flow_conservation(self):
        """Test flow conservation constraints for routing"""
        # Set order 1 and 2 to be in the batch
        self.x[1, 1].lb = 1
        self.x[2, 1].lb = 1
        self.x[3, 1].lb = 0

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE)

        for i in self.I:
            for b in self.B:
                if self.x[i, b].X > 0.5:  # If order i is in batch b
                    # Inflow equals 1 for this order (from depot or another order)
                    inflow = self.r_b[0, i, b].X + sum(self.r_b[j, i, b].X for j in self.I if j != i)
                    self.assertAlmostEqual(inflow, 1.0)

                    # Outflow equals 1 for this order (to depot or another order)
                    outflow = self.r_b[i, 0, b].X + sum(self.r_b[i, j, b].X for j in self.I if j != i)
                    self.assertAlmostEqual(outflow, 1.0)

    def test_route_starts_ends_at_depot(self):
        """Test that routes start and end at the depot if batch is non-empty"""
        # Set order 1 to be in the batch
        self.x[1, 1].lb = 1
        self.x[2, 1].lb = 0
        self.x[3, 1].lb = 0

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE)

        for b in self.B:
            has_orders = sum(self.x[i, b].X for i in self.I) > 0

            if has_orders:
                # At least one route starts from depot
                starts_from_depot = sum(self.r_b[0, j, b].X for j in self.I) > 0
                self.assertTrue(starts_from_depot)

                # At least one route ends at depot
                ends_at_depot = sum(self.r_b[j, 0, b].X for j in self.I) > 0
                self.assertTrue(ends_at_depot)

    def test_empty_batch_no_routing(self):
        """Test that empty batches have no routing"""
        # Set all orders to not be in the batch
        for i in self.I:
            self.x[i, 1].ub = 0

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE)

        for b in self.B:
            # Check that no routing variables are active
            total_routes = sum(self.r_b[i, j, b].X for i in self.N for j in self.N if i != j)
            self.assertAlmostEqual(total_routes, 0.0)


# Tests for No Self-Loop Constraint (18) and Subtour Elimination Constraints (19-22)
class TestNoLoopAndSubtour(unittest.TestCase):

    def setUp(self):
        self.model = Model("TestNoLoopAndSubtourModel")

        # Simplified test data
        self.I = [1, 2, 3]  # Orders
        self.B = [1]  # Single batch for simplicity
        self.N = [0, 1, 2, 3]  # Nodes: 0 = depot, 1-3 = orders
        self.M = 1000  # Large constant

        # Decision variables
        self.x = self.model.addVars(self.I, self.B, vtype=GRB.BINARY, name="x")  # Order to batch
        self.r_b = self.model.addVars(self.N, self.N, self.B, vtype=GRB.BINARY, name="r_b")  # Routing
        self.u_i = self.model.addVars(self.N, self.B, vtype=GRB.INTEGER, name="u_i")  # Subtour elimination

        # No loops constraint (18)
        self.model.addConstrs(
            (self.r_b[i, i, b] == 0 for b in self.B for i in self.N),
            name="NoSelfLoop"
        )

        # Subtour elimination in routing (19-22)
        self.model.addConstrs(
            (self.u_i[0, b] == 0 for b in self.B),
            name="DepotOrderPosition"
        )
        self.model.addConstrs(
            (self.u_i[i, b] + 1 <= self.u_i[j, b] + self.M * (1 - self.r_b[i, j, b])
             for b in self.B for i in self.N for j in self.I if i != j),
            name="RoutingSubtour1"
        )
        self.model.addConstrs(
            (self.u_i[i, b] <= self.M * quicksum(self.r_b[i, j, b] for j in self.N if j != i) for b in self.B for i in
             self.I),
            name="RoutingSubtour2"
        )
        self.model.addConstrs(
            (self.u_i[i, b] <= quicksum(self.x[k, b] for k in self.I) for b in self.B for i in self.I),
            name="RoutingSubtour3"
        )

        # Flow balance constraints for completeness
        self.model.addConstrs(
            (quicksum(self.r_b[j, i, b] for j in self.N if j != i) == self.x[i, b]
             for b in self.B for i in self.I),
            name="RoutingInflow"
        )
        self.model.addConstrs(
            (quicksum(self.r_b[i, j, b] for j in self.N if j != i) == self.x[i, b]
             for b in self.B for i in self.I),
            name="RoutingOutflow"
        )

        # Vehicle flow constraints (depot)
        self.model.addConstrs(
            (quicksum(self.r_b[0, j, b] for j in self.I) == 1 for b in self.B),
            name="DepotOutflow"
        )
        self.model.addConstrs(
            (quicksum(self.r_b[i, 0, b] for i in self.I) == 1 for b in self.B),
            name="DepotInflow"
        )

        # Objective function (to make model feasible)
        self.model.setObjective(quicksum(self.x[i, b] for i in self.I for b in self.B), GRB.MAXIMIZE)

        self.model.update()

    def reset_model_bounds(self):
        """Reset all variable bounds to their defaults"""
        for i in self.I:
            for b in self.B:
                self.x[i, b].lb = 0
                self.x[i, b].ub = 1

    def test_no_self_loops(self):
        """Test that there are no self-loops in the routing solution (constraint 18)"""
        self.reset_model_bounds()
        # Set at least one order to be in the batch to make the model feasible
        self.x[1, 1].lb = 1

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE)

        # Check that no node has a route to itself
        for b in self.B:
            for i in self.N:
                self.assertAlmostEqual(self.r_b[i, i, b].X, 0.0)

    def test_depot_position_zero(self):
        """Test that the depot's position is set to zero (constraint 19)"""
        self.reset_model_bounds()
        # Set at least one order to be in the batch to make the model feasible
        self.x[1, 1].lb = 1

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE)

        # Check that depot position is zero for all batches
        for b in self.B:
            self.assertAlmostEqual(self.u_i[0, b].X, 0.0)

    def test_sequential_routing_positions(self):
        """Test subtour elimination ordering constraints (constraints 20-22)"""
        self.reset_model_bounds()
        # Set orders 1 and 2 to be in the batch
        self.x[1, 1].lb = 1
        self.x[2, 1].lb = 1
        self.x[3, 1].lb = 0

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE, "Model is infeasible")

        for b in self.B:
            # Check if positions are consistent with routing decisions
            for i in self.N:
                for j in self.N:
                    if i != j and self.r_b[i, j, b].X > 0.5:  # If there's a route from i to j
                        # Position of j should be greater than position of i, except if j is depot (0)
                        if j != 0:  # Only check if j is not depot
                            self.assertGreater(self.u_i[j, b].X, self.u_i[i, b].X)

    def test_subtour_elimination_inactive_nodes(self):
        """Test that position variables are consistent for inactive nodes (constraints 21-22)"""
        self.reset_model_bounds()
        # Set order 1 to be in the batch
        self.x[1, 1].lb = 1
        self.x[2, 1].lb = 0
        self.x[3, 1].lb = 0

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE, "Model is infeasible")

        for b in self.B:
            # Count active orders in this batch
            active_orders = sum(self.x[k, b].X > 0.5 for k in self.I)

            for i in self.I:
                if self.x[i, b].X < 0.5:  # If order i is not in the batch
                    # Its position variable should be 0
                    self.assertLessEqual(self.u_i[i, b].X, 0.001)  # Allow for small numerical error
                else:
                    # For active orders, position should be between 1 and number of orders
                    self.assertGreaterEqual(self.u_i[i, b].X, 1)
                    self.assertLessEqual(self.u_i[i, b].X, active_orders)

    def test_complete_tour_formation(self):
        """Test that the constraints together ensure a valid tour without subtours"""
        self.reset_model_bounds()
        # Set all orders to be in the batch
        for i in self.I:
            self.x[i, 1].lb = 1

        self.model.optimize()

        # Check if model is feasible
        self.assertNotEqual(self.model.Status, GRB.INFEASIBLE, "Model is infeasible")

        batch = 1  # Testing for batch 1

        # Print model solution for debugging
        if self.model.Status == GRB.OPTIMAL:
            print("\nRouting variables:")
            for i in self.N:
                for j in self.N:
                    if i != j and self.r_b[i, j, batch].X > 0.5:
                        print(f"Route from {i} to {j}")

            print("\nPosition variables:")
            for i in self.N:
                print(f"Position of node {i}: {self.u_i[i, batch].X}")

        # Collect all edges in the solution
        edges = []
        for i in self.N:
            for j in self.N:
                if i != j and self.r_b[i, j, batch].X > 0.5:
                    edges.append((i, j))

        # Verify we have the right number of edges for a complete tour
        # In a tour with n nodes (including depot), we should have exactly n edges
        expected_edges = sum(self.x[i, batch].X > 0.5 for i in self.I)  # Active orders
        self.assertEqual(len(edges), expected_edges + 1)  # +1 for depot

        # Check if the tour is connected
        # Start from depot
        current = 0
        visited = {0}  # Use set for faster lookups
        tour = [0]

        # Try to follow the tour from node to node
        remaining_tries = len(self.N)  # Prevent infinite loop
        while len(visited) < expected_edges + 1 and remaining_tries > 0:
            next_node = None
            for i, j in edges:
                if i == current and j not in visited:
                    next_node = j
                    break

            if next_node is None:
                # If we can't find an unvisited node, see if we can return to depot
                if len(visited) == expected_edges + 1:
                    for i, j in edges:
                        if i == current and j == 0:
                            tour.append(0)
                            break
                    break
                self.fail(f"Disconnected tour detected! Current tour: {tour}")

            visited.add(next_node)
            tour.append(next_node)
            current = next_node
            remaining_tries -= 1

        # Check that we can return to depot
        found_return = False
        for i, j in edges:
            if i == current and j == 0 and len(visited) == expected_edges + 1:
                found_return = True
                if tour[-1] != 0:  # Only append if not already there
                    tour.append(0)
                break

        self.assertTrue(found_return, f"No return path to depot from last node {current}")

        # Verify tour length (should be number of active orders + 2 for depot at start and end)
        self.assertEqual(len(tour), expected_edges + 2)

# Tests for flow balance in scheduling (Constraints 23-24) and for nonempty vehicle schedule start/end (25-26)
# and no self loops in scheduling (27)
class TestFlowBalance(unittest.TestCase):
    def setUp(self):
        self.model = Model("TestModel")
        self.B = [1, 2, 3]  # Example batches
        self.V = ["V1", "V2"]  # Example vehicles
        self.B0 = [0] + self.B  # Include depot
        self.M = 1000  # Large constant

        # Decision variables
        self.s = self.model.addVars(self.B0, self.B0, self.V, vtype=GRB.BINARY, name="s")
        self.y = self.model.addVars(self.B, self.V, vtype=GRB.BINARY, name="y")

        # Flow balance constraints (23-24)
        self.model.addConstrs((
            self.s[0, i, v] + quicksum(self.s[j, i, v] for j in self.B if j != i) == self.y[i, v]
            for v in self.V for i in self.B
        ), name="SchedInflow")

        self.model.addConstrs((
            self.s[i, 0, v] + quicksum(self.s[i, j, v] for j in self.B if j != i) == self.y[i, v]
            for v in self.V for i in self.B
        ), name="SchedOutflow")

        # Nonempty vehicle schedule start/end (25-26)
        self.model.addConstrs((
            quicksum(self.y[i, v] for i in self.B) <= self.M * quicksum(self.s[0, j, v] for j in self.B)
            for v in self.V
        ), name="SchedStart")

        self.model.addConstrs((
            quicksum(self.y[i, v] for i in self.B) <= self.M * quicksum(self.s[j, 0, v] for j in self.B)
            for v in self.V
        ), name="SchedEnd")

        # No self loops in scheduling (27)
        self.model.addConstrs((
            self.s[b, b, v] == 0 for b in self.B0 for v in self.V
        ), name="NoSelfLoopScheduling")

        # Solve the model with dummy objective just for testing feasibility
        self.model.setObjective(0, GRB.MINIMIZE)
        self.model.optimize()

    def test_flow_balance(self):
        if self.model.status == GRB.OPTIMAL:
            for v in self.V:
                for i in self.B:
                    inflow = self.s[0, i, v].X + sum(self.s[j, i, v].X for j in self.B if j != i)
                    outflow = self.s[i, 0, v].X + sum(self.s[i, j, v].X for j in self.B if j != i)
                    self.assertAlmostEqual(inflow, self.y[i, v].X, places=5)
                    self.assertAlmostEqual(outflow, self.y[i, v].X, places=5)
        else:
            self.fail("Model did not solve to optimality")

    def test_nonempty_vehicle_schedule(self):
        if self.model.status == GRB.OPTIMAL:
            for v in self.V:
                total_y = sum(self.y[i, v].X for i in self.B)
                start_sum = sum(self.s[0, j, v].X for j in self.B)
                end_sum = sum(self.s[j, 0, v].X for j in self.B)
                self.assertLessEqual(total_y, self.M * start_sum)
                self.assertLessEqual(total_y, self.M * end_sum)
        else:
            self.fail("Model did not solve to optimality")

    def test_no_self_loop_scheduling(self):
        if self.model.status == GRB.OPTIMAL:
            for b in self.B0:
                for v in self.V:
                    self.assertEqual(self.s[b, b, v].X, 0)
        else:
            self.fail("Model did not solve to optimality")

# Tests for subtour elimination in scheduling (Constraints 28-31)
class TestSubtourEliminationConstraints(unittest.TestCase):
    def setUp(self):
        self.model = Model("TestModel")
        self.B = [1, 2, 3]  # Example batches
        self.V = ["V1", "V2"]  # Example vehicles
        self.B0 = [0] + self.B  # Include depot
        self.M = 1000  # Large constant

        # Decision variables
        self.s = self.model.addVars(self.B0, self.B0, self.V, vtype=GRB.BINARY, name="s")
        self.y = self.model.addVars(self.B, self.V, vtype=GRB.BINARY, name="y")
        self.u_bv = self.model.addVars(self.B0, self.V, vtype=GRB.CONTINUOUS, name="u_bv")

        # Subtour elimination constraints (28-31)
        self.model.addConstrs((self.u_bv[0, v] == 0 for v in self.V), name="SchedulingDepot")
        self.model.addConstrs((
            self.u_bv[i, v] + self.s[i, j, v] <= self.u_bv[j, v] + self.M * (1 - self.s[i, j, v])
            for v in self.V for i in self.B for j in self.B if i != j
        ), name="SchedulingSubtour1")
        self.model.addConstrs((
            self.u_bv[i, v] <= self.M * quicksum(self.s[i, j, v] for j in self.B if j != i)
            for v in self.V for i in self.B
        ), name="SchedulingSubtour2")
        self.model.addConstrs((
            self.u_bv[i, v] <= quicksum(self.y[j, v] for j in self.B)
            for v in self.V for i in self.B
        ), name="SchedulingSubtour3")

        # Solve the model with dummy objective just for testing feasibility
        self.model.setObjective(0, GRB.MINIMIZE)
        self.model.optimize()

    def test_depot_start(self):
        if self.model.status == GRB.OPTIMAL:
            for v in self.V:
                self.assertAlmostEqual(self.u_bv[0, v].X, 0, places=5)
        else:
            self.fail("Model did not solve to optimality")

    def test_subtour_elimination1(self):
        if self.model.status == GRB.OPTIMAL:
            for v in self.V:
                for i in self.B:
                    for j in self.B:
                        if i != j:
                            self.assertLessEqual(
                                self.u_bv[i, v].X + self.s[i, j, v].X,
                                self.u_bv[j, v].X + self.M * (1 - self.s[i, j, v].X)
                            )
        else:
            self.fail("Model did not solve to optimality")

    def test_subtour_elimination2(self):
        if self.model.status == GRB.OPTIMAL:
            for v in self.V:
                for i in self.B:
                    self.assertLessEqual(
                        self.u_bv[i, v].X,
                        self.M * sum(self.s[i, j, v].X for j in self.B if j != i)
                    )
        else:
            self.fail("Model did not solve to optimality")

    def test_subtour_elimination3(self):
        if self.model.status == GRB.OPTIMAL:
            for v in self.V:
                for i in self.B:
                    self.assertLessEqual(
                        self.u_bv[i, v].X,
                        sum(self.y[j, v].X for j in self.B)
                    )
        else:
            self.fail("Model did not solve to optimality")

if __name__ == "__main__":
    unittest.main()
