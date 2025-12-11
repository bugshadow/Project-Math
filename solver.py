import numpy as np
import pandas as pd

class SimplexSolver:
    def __init__(self, c, A, b):
        """
        Initialize the Simplex Solver.
        Maximize z = c^T * x
        Subject to A * x <= b
        x >= 0
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.steps = []
        self.status = "Initialized"

    def solve(self):
        # --- Step 1: Standard Form ---
        # Maximize Z = c.x
        # Subject to Ax + Is = b
        # x >= 0, s >= 0
        
        # Extended A matrix [A | I]
        A_std = np.hstack([self.A, np.eye(self.num_constraints)])
        
        # Extended c vector [c | 0]
        c_std = np.concatenate([self.c, np.zeros(self.num_constraints)])
        
        # Variables names
        var_names = [f'x{i+1}' for i in range(self.num_vars)] + \
                    [f's{i+1}' for i in range(self.num_constraints)]
        
        # --- Step 2: Initial Base Selection ---
        # We choose slack variables as the initial base
        # Base indices (0-based)
        basic_indices = list(range(self.num_vars, self.num_vars + self.num_constraints))
        non_basic_indices = list(range(self.num_vars))
        
        # Initial Base Matrix B (Identity)
        B = A_std[:, basic_indices]
        
        self.steps.append({
            "step_type": "initialization",
            "step_name": "Initialisation",
            "description": "Choix de la base initiale (variables d'écart).",
            "matrices": {
                "A_std": A_std,
                "c_std": c_std,
                "b": self.b,
                "B_init": B,
                "var_names": var_names,
                "basic_indices": basic_indices
            }
        })
        
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            # 1. Calculate B inverse
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                self.status = "Error: Singular Matrix"
                break
                
            # 2. Calculate X_b = B^-1 * b
            x_B = np.dot(B_inv, self.b)
            
            # 3. Calculate Simplex Multipliers (Dual variables) pi = c_B * B^-1
            # Get c_B (coefficients of basic vars in objective)
            c_B = c_std[basic_indices]
            pi = np.dot(c_B, B_inv)
            
            # 4. Calculate Reduced Costs (Z_j - C_j) for Non-Basic Variables
            # Z_j - C_j = pi * A_j - c_j
            # We want to find entering variable (most negative Z_j - C_j for maximization if we consider C_j - Z_j, 
            # BUT standard convention often uses Z_j - C_j. 
            # Let's stick to: We want to maximize. 
            # Improvement if (c_j - z_j) > 0  <=> (c_j - pi*A_j) > 0
            # Equivalently, if (z_j - c_j) < 0.
            # Let's calculate delta_j = z_j - c_j. We enter if delta_j < 0.
            
            deltas = {} # Store delta for each non-basic var
            entering_idx_rel = -1 # Index in non_basic_indices list
            min_delta = -1e-9 # Threshold for optimality
            
            for i, idx in enumerate(non_basic_indices):
                A_j = A_std[:, idx]
                z_j = np.dot(pi, A_j)
                c_j = c_std[idx]
                delta = z_j - c_j
                deltas[var_names[idx]] = delta
                
                if delta < min_delta:
                    min_delta = delta
                    entering_idx_rel = i
            
            # Current Objective Value
            z_val = np.dot(c_B, x_B)
            
            step_data = {
                "step_type": "iteration",
                "step_name": f"Itération {iteration + 1}",
                "iteration": iteration + 1,
                "matrices": {
                    "B": B,
                    "B_inv": B_inv,
                    "x_B": x_B,
                    "c_B": c_B,
                    "z_val": z_val,
                    "deltas": deltas
                },
                "basic_vars": [var_names[i] for i in basic_indices],
                "non_basic_vars": [var_names[i] for i in non_basic_indices]
            }
            
            # Check Optimality
            if entering_idx_rel == -1:
                self.status = "Optimal"
                step_data["description"] = "Tous les coûts réduits (Z_j - C_j) sont positifs ou nuls. Solution optimale atteinte."
                self.steps.append(step_data)
                break
                
            # Entering Variable
            entering_global_idx = non_basic_indices[entering_idx_rel]
            entering_var_name = var_names[entering_global_idx]
            
            # 5. Calculate Y = B^-1 * A_entering
            A_entering = A_std[:, entering_global_idx]
            Y = np.dot(B_inv, A_entering)
            
            step_data["matrices"]["Y"] = Y
            step_data["entering_var"] = entering_var_name
            
            # 6. Ratio Test (Minimum Ratio)
            # Ratio = x_B_i / Y_i for Y_i > 0
            ratios = []
            leaving_idx_rel = -1
            min_ratio = np.inf
            
            for i in range(self.num_constraints):
                if Y[i] > 1e-9:
                    ratio = x_B[i] / Y[i]
                    ratios.append(ratio)
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving_idx_rel = i
                else:
                    ratios.append(np.inf)
            
            step_data["ratios"] = ratios
            
            if leaving_idx_rel == -1:
                self.status = "Unbounded"
                step_data["description"] = "Problème non borné (tous les Y_i <= 0)."
                self.steps.append(step_data)
                break
                
            leaving_global_idx = basic_indices[leaving_idx_rel]
            leaving_var_name = var_names[leaving_global_idx]
            
            step_data["leaving_var"] = leaving_var_name
            step_data["pivot_element"] = Y[leaving_idx_rel]
            step_data["description"] = f"Variable entrante : {entering_var_name} (Z-C = {min_delta:.2f}). " \
                                       f"Variable sortante : {leaving_var_name} (Ratio = {min_ratio:.2f})."
            
            self.steps.append(step_data)
            
            # Update Base
            basic_indices[leaving_idx_rel] = entering_global_idx
            non_basic_indices[entering_idx_rel] = leaving_global_idx
            
            # Update B matrix
            B = A_std[:, basic_indices]
            
            iteration += 1
            
        if iteration == max_iterations:
            self.status = "Max Iterations Reached"

        # Extract Final Solution
        solution = np.zeros(self.num_vars)
        
        # Recalculate final x_B if we broke out at optimality
        if self.status == "Optimal":
            # B and basic_indices are already updated to optimal base? 
            # No, the loop breaks BEFORE updating if optimal.
            # So x_B from the last step is the optimal solution.
            pass
        
        # Map basic vars to solution vector
        # We need to be careful: x_B corresponds to basic_indices
        for i, idx in enumerate(basic_indices):
            if idx < self.num_vars: # If it's a decision variable (not slack)
                solution[idx] = x_B[i]
                
        return {
            "status": self.status,
            "max_profit": np.dot(self.c, solution),
            "solution": solution,
            "steps": self.steps
        }
