import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from itertools import combinations
from solver import SimplexSolver

# --- Page Configuration ---
st.set_page_config(
    page_title="Optimisation Cloud Computing",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Problem Box */
    .problem-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 5px solid #00d2ff;
        color: #e0e0e0;
    }
    
    /* Metrics */
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        text-align: center;
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00d2ff;
        color: #000000;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00b8e6;
        transform: scale(1.02);
        color: #000000;
    }
    
    /* Success/Info Messages */
    .stAlert {
        background-color: #262730;
        color: white;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def matrix_to_latex(mat, style='bmatrix'):
    """Converts a numpy array to a LaTeX matrix string."""
    if len(mat.shape) == 1:
        mat = mat.reshape(-1, 1) # Column vector by default
    
    lines = []
    for row in mat:
        row_strs = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                row_strs.append(f'{int(round(x))}')
            else:
                row_strs.append(f'{x:.2f}')
        lines.append(' & '.join(row_strs))
    
    content = ' \\\\ '.join(lines)
    return f"\\begin{{{style}}}\n{content}\n\\end{{{style}}}"

def get_vertices(A, b):
    """
    Calculate vertices of the feasible region defined by Ax <= b and x >= 0.
    """
    num_vars = 3
    # Add non-negativity constraints as -x <= 0, -y <= 0, -z <= 0
    A_full = np.vstack([A, -np.eye(num_vars)])
    b_full = np.concatenate([b, np.zeros(num_vars)])
    
    num_constraints = A_full.shape[0]
    vertices = []
    
    # Iterate through all combinations of 3 planes to find intersection points
    for indices in combinations(range(num_constraints), num_vars):
        try:
            A_sub = A_full[list(indices)]
            b_sub = b_full[list(indices)]
            
            # Solve linear system
            point = np.linalg.solve(A_sub, b_sub)
            
            # Check if point satisfies all other constraints
            if np.all(np.dot(A_full, point) <= b_full + 1e-7):
                vertices.append(point)
        except np.linalg.LinAlgError:
            continue

    if not vertices:
        return np.array([])
        
    return np.unique(np.array(vertices), axis=0)

# --- Main App ---

def main():
    st.title("☁️ Optimisation de Ressources Cloud")
    st.markdown("### Maximisation du profit par allocation optimale des tâches")

    # --- Problem Statement ---
    st.markdown("""
    <div class="problem-box">
        <h4>📋 Énoncé du Problème</h4>
        <p>Une entreprise de cloud computing exécute trois types de tâches (<b>CPU</b>, <b>RAM</b> et <b>GPU</b>). 
        L'objectif est de déterminer le nombre optimal de chaque tâche pour maximiser le profit total, 
        tout en respectant les contraintes de temps des ateliers de traitement.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar / Data Input ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        st.info("Modifiez les valeurs ci-dessous pour mettre à jour le modèle.")

    # Default Data
    default_data = {
        "Atelier": ["Tache 1", "Tache 2", "Tache 3"],
        "CPU (h)": [2.0, 1.0, 0.5],
        "RAM (h)": [1.0, 2.0, 0.5],
        "GPU (h)": [0.5, 0.5, 1.0],
        "Max. Temps (h)": [20.0, 20.0, 12.0]
    }
    df_constraints = pd.DataFrame(default_data)

    default_profits = {
        "Composant": ["CPU", "RAM", "GPU"],
        "Profit Unitaire (€)": [5.0, 10.0, 15.0]
    }
    df_profits = pd.DataFrame(default_profits)

    # Layout for inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Contraintes des Ateliers")
        edited_constraints = st.data_editor(df_constraints, num_rows="dynamic", use_container_width=True)
    
    with col2:
        st.subheader("2. Profits Unitaires")
        edited_profits = st.data_editor(df_profits, num_rows="fixed", use_container_width=True)

    # Extract Data for Solver
    try:
        # Constraints Matrix A and Vector b
        A = edited_constraints[["CPU (h)", "RAM (h)", "GPU (h)"]].values
        b = edited_constraints["Max. Temps (h)"].values
        
        # Objective Function Coefficients c
        c = edited_profits["Profit Unitaire (€)"].values
        
        # Validate dimensions
        if A.shape[1] != 3 or len(c) != 3:
            st.error("Le modèle doit avoir exactement 3 produits (A, B, C).")
            st.stop()
            
    except Exception as e:
        st.error(f"Erreur dans les données: {e}")
        st.stop()

    # --- Resolution ---
    
    solver = SimplexSolver(c, A, b)
    result = solver.solve()

    if result["status"] != "Optimal":
        st.warning(f"Attention: Le solveur n'a pas trouvé de solution optimale. Statut: {result['status']}")

    # --- Tabs for Visualization vs Algebra ---
    tab_graph, tab_algebra = st.tabs(["📊 Résolution Graphique (3D)", "🧮 Résolution Algébrique (Simplexe)"])

    with tab_graph:
        st.markdown("### Visualisation de la Zone Admissible et des Contraintes")
        
        vertices = get_vertices(A, b)
        
        fig = go.Figure()
        
        # 1. Plot Constraints as Planes
        # Define grid for x, y
        x_range = np.linspace(0, max(b)*1.2, 50)
        y_range = np.linspace(0, max(b)*1.2, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3'] # Plotly colors
        
        for i in range(len(b)):
            # Plane equation: a*x + b*y + c*z = d  =>  z = (d - a*x - b*y) / c
            # Handle division by zero if c=0
            if A[i][2] != 0:
                Z = (b[i] - A[i][0]*X - A[i][1]*Y) / A[i][2]
                
                # Clip Z to be non-negative for better visualization
                Z[Z < 0] = np.nan
                
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    opacity=0.3,
                    showscale=False,
                    colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]],
                    name=f"Contrainte {i+1} ({df_constraints['Atelier'][i]})",
                    colorbar=dict(len=0.5, y=0.5) # Hack to hide colorbar or manage it
                ))
                
                # Add a dummy trace for legend
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=colors[i % len(colors)]),
                    name=f"Contrainte {i+1}"
                ))

        # 2. Plot Feasible Region (Polyhedron)
        if len(vertices) >= 4:
            try:
                hull = ConvexHull(vertices)
                x, y, z = vertices.T
                
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=hull.simplices[:, 0],
                    j=hull.simplices[:, 1],
                    k=hull.simplices[:, 2],
                    opacity=0.2,
                    color='#ffffff',
                    name='Zone Admissible'
                ))
                
                # Plot Vertices
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(size=4, color='#ffffff'),
                    name='Sommets'
                ))
            except:
                pass

        # 3. Plot Optimal Point
        opt_sol = result["solution"]
        fig.add_trace(go.Scatter3d(
            x=[opt_sol[0]], y=[opt_sol[1]], z=[opt_sol[2]],
            mode='markers+text',
            marker=dict(size=10, color='#e74c3c', symbol='diamond'),
            text=["Optimal"],
            textposition="top center",
            name='Point Optimal'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis_title='CPU',
                yaxis_title='RAM',
                zaxis_title='GPU',
                xaxis=dict(backgroundcolor="#0e1117", gridcolor="#444", showbackground=True),
                yaxis=dict(backgroundcolor="#0e1117", gridcolor="#444", showbackground=True),
                zaxis=dict(backgroundcolor="#0e1117", gridcolor="#444", showbackground=True),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=700,
            paper_bgcolor="#0e1117",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Display Result Summary
        st.markdown("---")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Profit Max", f"{result['max_profit']:.2f} €")
        res_col2.metric("CPU", f"{result['solution'][0]:.2f}")
        res_col3.metric("RAM", f"{result['solution'][1]:.2f}")
        res_col4.metric("GPU", f"{result['solution'][2]:.2f}")

    with tab_algebra:
        # --- 1. Mise en équations ---
        st.markdown("## 1. Mise en équations (Modèle Linéaire)")
        
        st.markdown("""
        **Notations :**
        - $x_{CPU}$ = nombre de tâches CPU.
        - $x_{RAM}$ = nombre de tâches RAM.
        - $x_{GPU}$ = nombre de tâches GPU.
        """)
        
        # Construct LaTeX string dynamically
        obj_func = f"max Z = {c[0]} x_{{CPU}} + {c[1]} x_{{RAM}} + {c[2]} x_{{GPU}}"
        constraints_latex = ""
        for i in range(len(b)):
            constraints_latex += f"{A[i][0]} x_{{CPU}} + {A[i][1]} x_{{RAM}} + {A[i][2]} x_{{GPU}} \\le {b[i]} \\quad (\\text{{{df_constraints['Atelier'][i]}}}) \\\\"
        
        latex_str = f"""
        \\text{{Fonction objectif :}} \\\\
        {obj_func} \\\\
        \\text{{Contraintes :}} \\\\
        \\begin{{cases}}
        {constraints_latex}
        x_{{CPU}}, x_{{RAM}}, x_{{GPU}} \\ge 0
        \\end{{cases}}
        """
        st.latex(latex_str)
        
        st.markdown("### Sous forme matricielle")
        st.latex(f"""
        A = {matrix_to_latex(A)}, \\quad
        x = \\begin{{pmatrix}} x_{{CPU}} \\\\ x_{{RAM}} \\\\ x_{{GPU}} \\end{{pmatrix}}, \\quad
        b = {matrix_to_latex(b)}, \\quad
        c = {matrix_to_latex(c)}
        """)
        st.markdown(r"Le problème est : maximiser $c^T x$ sous la contrainte $A x \le b$, $x \ge 0$.")

        # --- 2. Résolution Algébrique ---
        st.markdown("## 2. Résolution Algébrique par Matrices")
        st.markdown("""
        L'optimum se trouve sur un **sommet** du polyèdre des contraintes. 
        Mathématiquement, cela correspond à la résolution d'un système linéaire $B x_B = b$ où $B$ est la matrice formée par les contraintes actives (ou la base optimale).
        """)
        
        # Get Optimal Basis from Solver Result
        # The solver stores steps. The last step (or the one with status Optimal) has the optimal basis.
        if result["status"] == "Optimal" and result["steps"]:
            last_step = result["steps"][-1]
            # If the last step is just an update without matrices, check the previous one or re-calculate
            # Actually solver stores matrices in "Selection" steps. The final state is after the last update.
            # Let's use the matrices from the last "Selection" step which led to optimality, 
            # OR better, let's reconstruct the final basis B from the final basic_vars.
            
            final_basic_vars = last_step["basic_vars"]
            
            # Reconstruct B for final basic vars
            # We need the full A matrix (including slacks) to extract columns
            num_vars = len(c)
            num_constraints = len(b)
            full_A = np.hstack([A, np.eye(num_constraints)])
            
            basic_indices = []
            for var in final_basic_vars:
                if var.startswith('x'):
                    basic_indices.append(int(var[1:]) - 1)
                else: # slack s_i
                    basic_indices.append(num_vars + int(var[1:]) - 1)
            
            B_opt = full_A[:, basic_indices]
            try:
                det_B = np.linalg.det(B_opt)
                B_inv_opt = np.linalg.inv(B_opt)
                x_B_opt = np.dot(B_inv_opt, b)
            except:
                det_B = 0
                B_inv_opt = np.zeros_like(B_opt)
                x_B_opt = np.zeros_like(b)

            st.markdown("### Calcul pour la Base Optimale")
            st.markdown(f"Les variables de base à l'optimum sont : **{', '.join(final_basic_vars)}**.")
            
            st.markdown("#### 1. Matrice de Base $B$")
            st.latex(f"B = {matrix_to_latex(B_opt)}")
            
            st.markdown("#### 2. Déterminant de $B$")
            st.latex(f"\\det(B) = {det_B:.2f}")
            if abs(det_B) > 1e-9:
                st.markdown(f"Puisque $\\det(B) \\neq 0$, la matrice est inversible.")
            
            st.markdown("#### 3. Inverse de $B$ ($B^{-1}$)")
            st.latex(f"B^{{-1}} = {matrix_to_latex(B_inv_opt)}")
            
            st.markdown("#### 4. Calcul de la solution $x_B = B^{-1} b$")
            st.latex(f"""
            x_B = {matrix_to_latex(B_inv_opt)} {matrix_to_latex(b)} = {matrix_to_latex(x_B_opt)}
            """)
            
            st.markdown("#### Résultat")
            
            # Helper to format float or int
            def fmt(x):
                return f"{int(round(x))}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"
            
            st.markdown(f"""
            Le point optimal est donc :
            - **CPU ($x_1$)** : {fmt(result['solution'][0])}
            - **RAM ($x_2$)** : {fmt(result['solution'][1])}
            - **GPU ($x_3$)** : {fmt(result['solution'][2])}
            
            **Profit Maximum** : {fmt(result['max_profit'])} €
            """)

        # --- 3. Lien avec le Simplexe ---
        st.markdown("## 3. Lien avec la méthode du Simplexe")
        st.info("""
        **Lien explicite avec l'algèbre linéaire :**
        *   Le système $B x_B = b$ représente l'intersection des contraintes actives.
        *   L'algorithme du simplexe itère de sommet en sommet (en changeant la base $B$) jusqu'à trouver celui qui maximise $Z$.
        *   À chaque étape, on calcule $x_B = B^{-1} b$ et on vérifie si on peut améliorer le profit.
        """)

        with st.expander("Voir les itérations détaillées du Simplexe (Tableaux)"):
             for step in result["steps"]:
                st.markdown(f"**{step['step_name']}**")
                st.markdown(f"_{step['description']}_")
                
                # Convert Tableau to LaTeX
                tableau = step["tableau"]
                formatted_tableau = [[f"{x:.2f}" for x in row] for row in tableau]
                num_cols = tableau.shape[1]
                col_format = "c" * (num_cols - 1) + "|c"
                
                latex_rows = []
                header_row = step["headers"]
                latex_rows.append(" & ".join([f"\\text{{{h}}}" for h in header_row]))
                latex_rows.append("\\hline")
                
                for i, row in enumerate(formatted_tableau):
                    latex_rows.append(" & ".join(row))
                    if i == len(formatted_tableau) - 2: 
                        latex_rows.append("\\hline")
                
                latex_content = " \\\\ ".join(latex_rows)
                
                st.latex(f"\\begin{{array}}{{{col_format}}} {latex_content} \\end{{array}}")
                
                if step["pivot"]:
                    st.info(f"📍 Pivot: Ligne {step['pivot'][0]+1}, Colonne {step['headers'][step['pivot'][1]]}")
                st.divider()

if __name__ == "__main__":
    main()
