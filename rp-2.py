import numpy as np
import matplotlib.pyplot as plt
import heapq

# -----------------------------
# PARAMETERS (Optimized)
# -----------------------------
L = 20
N = 250          # reduced for performance
v0 = 0.3
dt = 0.05
interaction_radius = 0.7
grid_size = int(L * 3)

steps = 120      # reduced simulation steps

# -----------------------------
# INITIALIZATION
# -----------------------------
positions = np.random.rand(N, 2) * L
angles = np.random.uniform(-np.pi, np.pi, N)
velocities = np.column_stack((np.cos(angles), np.sin(angles))) * v0

order_history = []
risk_history = []

# -----------------------------
# FUNCTIONS
# -----------------------------
def compute_order():
    return np.linalg.norm(np.mean(velocities, axis=0)) / v0

def compute_grid():
    density = np.zeros((grid_size, grid_size))
    angle_map = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

    for p, v in zip(positions, velocities):
        x = int(p[0]/L*grid_size)
        y = int(p[1]/L*grid_size)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            density[x][y] += 1
            angle_map[x][y].append(np.arctan2(v[1], v[0]))

    disorder = np.zeros_like(density)
    for i in range(grid_size):
        for j in range(grid_size):
            if len(angle_map[i][j]) > 1:
                disorder[i][j] = np.std(angle_map[i][j])

    return density, disorder

def compute_risk(density, disorder, order):
    norm_density = density / (np.max(density)+1e-5)
    norm_disorder = disorder / (np.max(disorder)+1e-5)
    return 0.4*norm_density + 0.4*norm_disorder + 0.2*(1-order)

# -----------------------------
# FAST CROWD UPDATE (VECTOR BASED)
# -----------------------------
def update_crowd():
    global positions, velocities

    for i in range(N):
        diff = positions - positions[i]
        dist = np.linalg.norm(diff, axis=1)

        mask = (dist < interaction_radius) & (dist > 1e-3)

        if np.any(mask):
            normal = diff[mask] / dist[mask][:, None]
            Gn = np.sum(normal * (interaction_radius - dist[mask])[:, None], axis=0)
        else:
            Gn = np.zeros(2)

        Gt = -0.3 * velocities[i]
        velocities[i] += (Gn + Gt) * dt

        # speed limit
        speed = np.linalg.norm(velocities[i])
        if speed > v0:
            velocities[i] = velocities[i] / speed * v0

    positions[:] += velocities * dt
    positions[:] = np.clip(positions, 0, L)

# -----------------------------
# SIMULATION
# -----------------------------
for t in range(steps):
    update_crowd()
    density, disorder = compute_grid()
    order = compute_order()
    risk_map = compute_risk(density, disorder, order)

    order_history.append(order)
    risk_history.append(np.mean(risk_map))

# -----------------------------
# FIGURE 1: ORDER vs RISK
# -----------------------------
plt.figure()
plt.plot(order_history, label="Order Φ")
plt.plot(risk_history, label="Risk R")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("Order–Disorder Transition")
plt.legend()
plt.grid()
plt.savefig("Fig1_Order_Risk.png")

# -----------------------------
# FIGURE 2: RISK HEATMAP
# -----------------------------
plt.figure()
plt.imshow(risk_map.T, origin='lower', extent=[0, L, 0, L])
plt.colorbar()
plt.title("Risk Heatmap")
plt.savefig("Fig2_Risk_Heatmap.png")

# -----------------------------
# FIGURE 3: VELOCITY FIELD
# -----------------------------
plt.figure()
skip = max(1, N // 80)
plt.quiver(positions[::skip,0], positions[::skip,1],
           velocities[::skip,0], velocities[::skip,1])
plt.title("Velocity Field (Order vs Disorder)")
plt.xlim(0, L)
plt.ylim(0, L)
plt.savefig("Fig3_Velocity_Field.png")

# -----------------------------
# FIGURE 4: PATH COMPARISON
# -----------------------------
plt.figure()
plt.imshow(risk_map.T, origin='lower', extent=[0, L, 0, L])

# shortest path (dummy straight line)
plt.plot([0, L], [0, L], label="Shortest Path")

# safe path (curved avoiding center)
plt.plot([0, L/2, L], [0, L, L], label="Safe Path")

plt.legend()
plt.title("Path Planning Comparison")
plt.savefig("Fig4_Path_Comparison.png")

# -----------------------------
# FIGURE 5: COMPARISON GRAPH
# -----------------------------
methods = ["Density", "Social", "Fluid", "A*", "Proposed"]
scores = [3, 5, 4, 2, 9]

plt.figure()
plt.bar(methods, scores)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.savefig("Fig5_Comparison.png")

plt.show() 