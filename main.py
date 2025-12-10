import numpy as np

# transposed to Rows for easier calculation
X = np.array([
    [-1, -1, -1, 0, -1, 1],  # x1
    [1, -1, 1, 1, 2, 1],     # x2
    [-1, 0, 2, 1, -1, 2],    # x3
    [-2, -1, -2, 1, -2, 1]   # x4
]).T 

V = np.array([
    [-0.9, 1.3, 0.4, -1.0, 0.7, 1.8],
    [1.2, 0.4, 0.9, -0.2, 0.1, -1.2],
    [-1.2, -0.4, 0.7, -2.0, -1.7, 0.9],
    [-0.4, -0.4, 1.7, 0.4, -2.1, -0.2]
]).T

# Local Best (P)
P = np.array([
    [0, -1, 1, -1, 0, -2],
    [-2, -1, 1, 0, -2, -1],
    [1, 0, -2, 1, 1, -1],
    [0, 0, -2, 1, 1, 0]
]).T

# Global Best (Pg)
Pg = np.array([-2, -1, -1, 0])

c1 = 0.9  # Phi 1
c2 = 0.4  # Phi 2
w = 1.0   # Inertia weight (assumed standard for basic PSO)
r1 = 0.5
r2 = 0.5

# Cost Function: 5x1^3 + 3x2^4 + 8x3^5 + 4x4^6
def cost_function(pos):
    return 5*(pos[0]**3) + 3*(pos[1]**4) + 8*(pos[2]**5) + 4*(pos[3]**6)

for iteration in range(1, 5):
    print(f"Iteration {iteration}")
    for i in range(len(X)):
        # V_new = w*V + c1*r1*(P_local - X) + c2*r2*(P_global - X)
        cognitive = c1 * r1 * (P[i] - X[i])
        social = c2 * r2 * (Pg - X[i])
        V[i] = (w * V[i]) + cognitive + social
        
        # Position Update
        X[i] = X[i] + V[i]
        
        #Fitness
        current_cost = cost_function(X[i])
        pbest_cost = cost_function(P[i])
        
        # Local Best
        if current_cost < pbest_cost:
            P[i] = X[i].copy()
            
        # Global Best
        if current_cost < cost_function(Pg):
            Pg = X[i].copy()

        # output 
        v_str = f"[{V[i][0]:.1f}, {V[i][1]:.1f}, {V[i][2]:.1f}, {V[i][3]:.1f}]"
        x_str = f"[{X[i][0]:.1f}, {X[i][1]:.1f}, {X[i][2]:.1f}, {X[i][3]:.1f}]"
        print(f"P{i+1:<4} Velocities: {v_str}  X: {x_str} Cost: {current_cost:.1f}")

    print(f"Global Best after Iter {iteration}: {Pg} Cost: {cost_function(Pg)}")
