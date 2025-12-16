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

    def solve_tableau(self):
        """
        Solves the LP using the full Tableau method.
        Returns a list of steps, where each step contains the tableau state.
        (Force Update)
        """
        steps = []
        
        # --- 1. Initialization ---
        # Variables: x1..xn, s1..sm
        num_vars = self.num_vars
        num_slack = self.num_constraints
        total_cols = num_vars + num_slack
        
        # Create Initial Tableau
        # Structure: Rows = Constraints + Z (last row)
        # Cols = x vars + s vars + RHS
        
        tableau = np.zeros((self.num_constraints + 1, total_cols + 1))
        
        # Fill Constraints (A matrix)
        tableau[:self.num_constraints, :num_vars] = self.A
        
        # Fill Slack Variables (Identity)
        tableau[:self.num_constraints, num_vars:total_cols] = np.eye(self.num_slack if hasattr(self, 'num_slack') else self.num_constraints)
        
        # Fill RHS (b vector)
        tableau[:self.num_constraints, -1] = self.b
        
        # Fill Z Row
        # Max Z = cx  =>  Z - cx = 0
        # So coefficients are -c for x vars, 0 for slack
        tableau[-1, :num_vars] = -self.c
        
        # Variable Names
        col_headers = [f'x{i+1}' for i in range(num_vars)] + \
                      [f's{i+1}' for i in range(self.num_constraints)] + \
                      ['RHS']
        
        row_headers = [f's{i+1}' for i in range(self.num_constraints)] + ['Z'] # Initial basis is slack
        
        steps.append({
            "step_id": 0,
            "description": "Tableau Initial",
            "tableau": tableau.copy(),
            "headers": col_headers,
            "basic_vars": row_headers[:] # Copy list
        })
        
        iteration = 0
        max_iter = 100
        
        while iteration < max_iter:
            # --- 2. Check Optimality ---
            # Look for most negative value in Z row (Last row, excluding RHS)
            z_row = tableau[-1, :-1]
            min_val = np.min(z_row)
            
            if min_val >= -1e-9:
                steps.append({
                    "step_id": iteration + 1,
                    "description": "Solution Optimale Trouvée (Tous coeff Z >= 0)",
                    "tableau": tableau.copy(),
                    "headers": col_headers,
                    "status": "Optimal"
                })
                break
                
            # --- 3. Pivot Column (Entering Variable) ---
            pivot_col_idx = np.argmin(z_row)
            entering_var = col_headers[pivot_col_idx]
            
            # --- 4. Pivot Row (Leaving Variable) ---
            # Ratio test: RHS / Column coeff, for coeff > 0
            ratios = []
            valid_rows = []
            
            min_ratio = np.inf
            pivot_row_idx = -1
            
            for i in range(self.num_constraints):
                col_val = tableau[i, pivot_col_idx]
                rhs_val = tableau[i, -1]
                
                if col_val > 1e-9:
                    ratio = rhs_val / col_val
                    ratios.append(ratio)
                    valid_rows.append(i)
                    
                    if ratio < min_ratio:
                        min_ratio = ratio
                        pivot_row_idx = i
                else:
                    ratios.append(np.inf)
            
            if pivot_row_idx == -1:
                steps.append({
                    "step_id": iteration + 1,
                    "description": "Problème non borné",
                    "tableau": tableau.copy(),
                    "headers": col_headers,
                    "status": "Unbounded"
                })
                break
            
            leaving_var = row_headers[pivot_row_idx]
            pivot_element = tableau[pivot_row_idx, pivot_col_idx]
            
            # Record step before operation
            steps[-1]["pivot_info"] = {
                "entering_var": entering_var,
                "leaving_var": leaving_var,
                "pivot_row": pivot_row_idx,
                "pivot_col": pivot_col_idx,
                "pivot_element": pivot_element,
                "ratios": ratios
            }
            
            # --- 5. Pivot Operation (Gaussian Elimination) ---
            new_tableau = tableau.copy()
            
            # Normalize Pivot Row
            new_tableau[pivot_row_idx, :] = tableau[pivot_row_idx, :] / pivot_element
            
            # Update other rows
            for i in range(tableau.shape[0]):
                if i != pivot_row_idx:
                    factor = tableau[i, pivot_col_idx]
                    new_tableau[i, :] = tableau[i, :] - factor * new_tableau[pivot_row_idx, :]
            
            tableau = new_tableau
            
            # Update Basis
            row_headers[pivot_row_idx] = entering_var
            
            steps.append({
                "step_id": iteration + 1,
                "description": f"Itération {iteration+1}",
                "tableau": tableau.copy(),
                "headers": col_headers,
                "basic_vars": row_headers[:]
            })
            
            iteration += 1
            
        return steps

