import numpy as np
from solver import SimplexSolver

def test_tableau_solver():
    print("Testing Tableau Solver...")
    
    # Test Data (from app default)
    # Max Z = 5x1 + 10x2 + 15x3
    # 2x1 + x2 + 0.5x3 <= 20
    # x1 + 2x2 + 0.5x3 <= 20
    # 0.5x1 + 0.5x2 + 1x3 <= 12
    
    A = [
        [2.0, 1.0, 0.5],
        [1.0, 2.0, 0.5],
        [0.5, 0.5, 1.0]
    ]
    b = [20.0, 20.0, 12.0]
    c = [5.0, 10.0, 15.0]
    
    solver = SimplexSolver(c, A, b)
    
    # Run Tableau Solve
    steps = solver.solve_tableau()
    
    print(f"Number of steps: {len(steps)}")
    
    # Get last step
    last_step = steps[-1]
    print(f"Final Status: {last_step.get('status', 'Unknown')}")
    
    # Check Result from Standard Solve (Revised Simplex)
    res_std = solver.solve()
    print(f"Revised Simplex Max Profit: {res_std['max_profit']}")
    
    # Extract Max Profit from Tableau
    # In Tableau, bottom-right cell usually contains -Z (or Z depending on formulation)
    # My initialization was Z - cx = 0.
    # Row operations maintain this eq.
    # At opt: Z + ... = RHS.
    # So RHS of Z row should be Z_max.
    
    tableau = last_step['tableau']
    z_max_tableau = tableau[-1, -1]
    print(f"Tableau Simplex Max Profit (RHS of Z row): {z_max_tableau}")
    
    if abs(z_max_tableau - res_std['max_profit']) < 1e-5:
        print("SUCCESS: Results match!")
    else:
        print("FAILURE: Results do not match.")
        
if __name__ == "__main__":
    test_tableau_solver()
