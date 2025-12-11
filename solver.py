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
        # 1. Convert to Standard Form (Add Slack Variables)
        # Tableau structure:
        # [ A | I | b ]
        # [ -c | 0 | 0 ]
        
        tableau = np.zeros((self.num_constraints + 1, self.num_vars + self.num_constraints + 1))
        
        # Fill A
        tableau[:self.num_constraints, :self.num_vars] = self.A
        
        # Fill Identity (Slack variables)
        tableau[:self.num_constraints, self.num_vars:self.num_vars + self.num_constraints] = np.eye(self.num_constraints)
        
        # Fill b
        tableau[:self.num_constraints, -1] = self.b
        
        # Fill c (Objective function row, negated for maximization)
        tableau[-1, :self.num_vars] = -self.c
        
        # Column headers
        col_headers = [f'x{i+1}' for i in range(self.num_vars)] + \
                      [f's{i+1}' for i in range(self.num_constraints)] + ['RHS']
        
        # Row headers (Basic variables)
        # Initially, slack variables are basic
        basic_vars = [f's{i+1}' for i in range(self.num_constraints)]
        
        self.steps.append({
            "step_name": "Initial Tableau",
            "tableau": tableau.copy(),
            "headers": col_headers,
            "basic_vars": basic_vars.copy(),
            "pivot": None,
            "description": "Tableau initial avec variables d'écart ajoutées."
        })
        
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            # Check for optimality: if all coefficients in the bottom row are >= 0
            if np.all(tableau[-1, :-1] >= -1e-9):
                self.status = "Optimal"
                break
                
            # Select Pivot Column (Entering Variable) - Most negative coefficient
            pivot_col = np.argmin(tableau[-1, :-1])
            entering_var = col_headers[pivot_col]
            
            # Select Pivot Row (Leaving Variable) - Minimum Ratio Test
            # ratio = b / column_val for column_val > 0
            ratios = []
            for i in range(self.num_constraints):
                val = tableau[i, pivot_col]
                if val > 1e-9:
                    ratios.append(tableau[i, -1] / val)
                else:
                    ratios.append(np.inf)
            
            if np.all(np.array(ratios) == np.inf):
                self.status = "Unbounded"
                break
                
            pivot_row = np.argmin(ratios)
            leaving_var = basic_vars[pivot_row]
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Store Matrix Details for this step
            # B = Matrix of basic columns from original A (augmented with slack)
            # We need to reconstruct the full initial matrix to extract B correctly
            # Initial Tableau (without objective row) contains [A | I]
            # Let's reconstruct the full constraint matrix for reference
            full_A = np.hstack([self.A, np.eye(self.num_constraints)])
            
            # Identify indices of basic variables
            basic_indices = []
            for var in basic_vars:
                if var.startswith('x'):
                    basic_indices.append(int(var[1:]) - 1)
                else: # slack s_i
                    basic_indices.append(self.num_vars + int(var[1:]) - 1)
            
            B = full_A[:, basic_indices]
            try:
                B_inv = np.linalg.inv(B)
            except:
                B_inv = np.zeros_like(B) # Should not happen in Simplex unless degenerate/error

            # C_B (Coefficients of basic vars in objective function)
            # Note: Slack vars have 0 coefficient
            c_B = np.zeros(self.num_constraints)
            for i, idx in enumerate(basic_indices):
                if idx < self.num_vars:
                    c_B[i] = self.c[idx]
                else:
                    c_B[i] = 0
            
            # X_B = B^-1 * b
            x_B = np.dot(B_inv, self.b)
            
            # Z = c_B * x_B
            z_val = np.dot(c_B, x_B)

            self.steps.append({
                "step_name": f"Iteration {iteration + 1} - Selection",
                "tableau": tableau.copy(),
                "headers": col_headers,
                "basic_vars": basic_vars.copy(),
                "pivot": (pivot_row, pivot_col),
                "matrices": {
                    "B": B,
                    "B_inv": B_inv,
                    "c_B": c_B,
                    "x_B": x_B,
                    "z_val": z_val
                },
                "description": f"Variable entrante: {entering_var} (coeff: {tableau[-1, pivot_col]:.2f}). "
                               f"Variable sortante: {leaving_var} (ratio: {ratios[pivot_row]:.2f}). "
                               f"Pivot: {pivot_element:.2f}."
            })
            
            # Perform Pivot Operation
            # 1. Normalize the pivot row
            tableau[pivot_row, :] /= pivot_element
            
            # 2. Eliminate other rows
            for i in range(self.num_constraints + 1):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    tableau[i, :] -= factor * tableau[pivot_row, :]
            
            # Update Basic Variables
            basic_vars[pivot_row] = entering_var
            
            self.steps.append({
                "step_name": f"Iteration {iteration + 1} - Update",
                "tableau": tableau.copy(),
                "headers": col_headers,
                "basic_vars": basic_vars.copy(),
                "pivot": None,
                "matrices": None, # Matrices are relevant before pivot or we can calc them again, but selection step is best for B/B_inv display
                "description": f"Tableau mis à jour après pivot sur {entering_var} et {leaving_var}."
            })
            
            iteration += 1
            
        if iteration == max_iterations:
            self.status = "Max Iterations Reached"

        # Extract Solution
        solution = np.zeros(self.num_vars)
        for i, var in enumerate(basic_vars):
            if var.startswith('x'):
                idx = int(var[1:]) - 1
                solution[idx] = tableau[i, -1]
                
        max_profit = tableau[-1, -1]
        
        return {
            "status": self.status,
            "max_profit": max_profit,
            "solution": solution,
            "steps": self.steps
        }
