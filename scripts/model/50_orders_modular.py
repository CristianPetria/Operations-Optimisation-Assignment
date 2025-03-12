from gurobipy import GRB
import logging

# Import shared functions from model_core
from core_model import (
    create_base_settings,
    create_optimization_model,
    run_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Callback function to monitor variable 'a'
def a_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # Retrieve the objective value of the new solution
        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        print(f"\nNew feasible solution found with objective value: {obj_val}")
        
        # Get the I set from model's private data
        I = model._I
        
        # Retrieve the current solution for variable 'a'
        a_vars = [model.getVarByName(f"a[{i}]") for i in I]
        a_values = model.cbGetSolution(a_vars)
        
        # Identify which orders are accepted
        accepted_orders = [i for i, val in zip(I, a_values) if val > 0.5]
        rejected_orders = [i for i, val in zip(I, a_values) if val <= 0.5]
        
        print(f"Accepted orders: {accepted_orders}")
        print(f"Rejected orders: {rejected_orders}")

def run_with_callback(settings, time_limit=300):
    """
    Run the model with a callback to monitor the accepted orders.
    
    Args:
        settings (dict): Dictionary containing all model parameters
        time_limit (int): Time limit in seconds
    
    Returns:
        tuple: (model, variables) - The optimized model and its variables
    """
    # Create the model and get variables
    model, variables = create_optimization_model(settings)
    
    # Set parameters
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.Presolve, 0)
    model.setParam(GRB.Param.OutputFlag, 1)
    model.setParam(GRB.Param.LogFile, "gurobi_log.txt")
    
    # Store I in the model for the callback
    model._I = settings["I"]
    
    # Optimize with the callback
    model.optimize(a_callback)
    
    # Check for infeasibility and compute IIS if necessary
    if model.status == GRB.INFEASIBLE:
        logger.error("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model.ilp")
        for c in model.getConstrs():
            if c.IISConstr:
                logger.error(f"IIS Constraint: {c.constrName}")
        logger.error("IIS written to model.ilp")
    
    # Print final solution if optimal
    if model.status == GRB.OPTIMAL:
        logger.info(f"\nOptimal objective value: {model.objVal}")
        logger.info("Final accepted orders:")
        
        # Extract variable a
        a = variables["a"]
        I = settings["I"]
        
        accepted = 0
        rejected = 0
        
        for i in I:
            if a[i].X > 0.5:
                logger.info(f"Order {i}: Accepted")
                accepted += 1
            else:
                logger.info(f"Order {i}: Rejected")
                rejected += 1
        
        logger.info(f"Total accepted orders: {accepted}")
        logger.info(f"Total rejected orders: {rejected}")
        logger.info(f"Acceptance rate: {accepted / len(I) * 100:.2f}%")
    
    return model, variables

if __name__ == "__main__":
    # Create settings for a medium-sized problem
    settings = create_base_settings(num_orders=20)
    
    # Adjust any specific settings if needed
    settings["CA"] = 30  # Increase capacity
    
    # Run the model with the callback
    logger.info("Starting optimization with callback...")
    model, variables = run_with_callback(settings, time_limit=300)