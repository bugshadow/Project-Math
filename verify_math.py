import numpy as np
from scipy.optimize import linprog
from solver import SimplexSolver

def verify():
    # Problem Data
    c = [5, 10, 15] # Maximize 5x + 10y + 15z
    # Scipy minimizes, so we minimize -c
    c_scipy = [-5, -10, -15]
    
    A = [
        [2, 1, 0.5],
        [1, 2, 0.5],
        [0.5, 0.5, 1]
    ]
    b = [20, 20, 12]
    
    # 1. Run Scipy (Ground Truth)
    res_scipy = linprog(c_scipy, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    
    print("--- Scipy Result ---")
    print(f"Status: {res_scipy.message}")
    print(f"X: {res_scipy.x}")
    print(f"Max Profit: {-res_scipy.fun}")
    
    # 2. Run Custom Solver
    solver = SimplexSolver(c, A, b)
    res_custom = solver.solve()
    
    print("\n--- Custom Solver Result ---")
    print(f"Status: {res_custom['status']}")
    print(f"X: {res_custom['solution']}")
    print(f"Max Profit: {res_custom['max_profit']}")
    
    # 3. Check Basis Reconstruction Logic
    print("\n--- Basis Check ---")
    if res_custom['steps']:
        last_step = res_custom['steps'][-1]
        basic_vars = last_step['basic_vars']
        print(f"Final Basic Vars: {basic_vars}")
        
        # Reconstruct B
        # Full A: [A | I]
        full_A = np.hstack([A, np.eye(3)])
        
        basic_indices = []
        for var in basic_vars:
            if var.startswith('x'):
                basic_indices.append(int(var[1:]) - 1)
            else: # s_i
                basic_indices.append(3 + int(var[1:]) - 1)
        
        B = full_A[:, basic_indices]
        print("Matrix B:")
        print(B)
        
        try:
            B_inv = np.linalg.inv(B)
            x_B = np.dot(B_inv, b)
            print("Calculated x_B (from B^-1 * b):", x_B)
            
            # Calculate Profit from x_B
            # We need to map x_B back to x1, x2, x3 to calculate profit, 
            # or use c_B * x_B
            
            # c_B construction
            c_B = []
            for var in basic_vars:
                if var.startswith('x'):
                    idx = int(var[1:]) - 1
                    c_B.append(c[idx])
                else:
                    c_B.append(0)
            c_B = np.array(c_B)
            
            z_val = np.dot(c_B, x_B)
            print(f"Calculated Z (c_B * x_B): {z_val}")
            
        except Exception as e:
            print(f"Error in matrix calc: {e}")

if __name__ == "__main__":
    verify()
