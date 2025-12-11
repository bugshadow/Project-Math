import numpy as np
from solver import SimplexSolver

def test_solver():
    # Problem:
    # Max Z = 5x1 + 10x2 + 15x3
    # Subject to:
    # 2x1 + 1x2 + 0.5x3 <= 20
    # 1x1 + 2x2 + 0.5x3 <= 20
    # 0.5x1 + 0.5x2 + 1x3 <= 12
    
    c = [5, 10, 15]
    A = [
        [2, 1, 0.5],
        [1, 2, 0.5],
        [0.5, 0.5, 1]
    ]
    b = [20, 20, 12]
    
    solver = SimplexSolver(c, A, b)
    result = solver.solve()
    
    print(f"Status: {result['status']}")
    print(f"Max Profit: {result['max_profit']}")
    print(f"Solution: {result['solution']}")
    
    print("\nSteps:")
    for step in result["steps"]:
        print(f"--- {step['step_name']} ---")
        print(f"Description: {step['description']}")
        if "matrices" in step:
            mats = step["matrices"]
            if "B_inv" in mats:
                print("B_inv shape:", mats["B_inv"].shape)
            if "deltas" in mats:
                print("Deltas:", mats["deltas"])
            if "Y" in mats:
                print("Y:", mats["Y"])
                
    # Expected Result:
    # Optimal solution should be around x1=0, x2=8, x3=8 (approx check)
    # 2(0) + 8 + 4 = 12 <= 20
    # 0 + 16 + 4 = 20 <= 20
    # 0 + 4 + 8 = 12 <= 12
    # Profit = 5(0) + 10(8) + 15(8) = 80 + 120 = 200
    
    assert result["status"] == "Optimal"
    assert np.isclose(result["max_profit"], 200.0)
    print("\nVerification Successful!")

if __name__ == "__main__":
    test_solver()
