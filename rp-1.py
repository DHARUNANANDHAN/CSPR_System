import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# -----------------------------
# INPUT PARAMETERS
# -----------------------------
L = float(input("Enter area size (meters): "))
CONGESTION_LEVEL = int(input("Enter congestion level (1=low,2=medium,3=high): "))

if CONGESTION_LEVEL == 1:
    N = 250; interaction_radius = 0.8
elif CONGESTION_LEVEL == 2:
    N = 400; interaction_radius = 0.7
else:
    N = 550; interaction_radius = 0.6

v0 = 0.3
dt = 0.05
grid_size = int(L * 3)

# -----------------------------
# INITIALIZATION
# -----------------------------
positions = np.random.rand(N, 2) * L
angles = np.random.uniform(-np.pi, np.pi, N)
velocities = np.column_stack((np.cos(angles), np.sin(angles))) * v0

# Path planning
target_point = None
start_point = None

# History
order_history = []
risk_history = []
comparison_history = {"density": [], "social": [], "fluid": [], "proposed": []}

# -----------------------------
# CROWD PARAMETERS
# -----------------------------
def compute_order():
    return np.linalg.norm(np.mean(velocities, axis=0)) / v0

def compute_grid():
    density = np.zeros((grid_size, grid_size))
    angle_map = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

    for p, v in zip(positions, velocities):
        x = int(p[0] / L * grid_size)
        y = int(p[1] / L * grid_size)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            density[x][y] += 1
            angle_map[x][y].append(np.arctan2(v[1], v[0]))

    disorder = np.zeros_like(density)
    for i in range(grid_size):
        for j in range(grid_size):
            if len(angle_map[i][j]) > 1:
                disorder[i][j] = np.std(angle_map[i][j])
    return density, disorder

# -----------------------------
# RISK MODELS (COMPARISON)
# -----------------------------
def compute_risk_model(density, disorder, order, mode):
    norm_d = density / (np.max(density) + 1e-5)
    norm_s = disorder / (np.max(disorder) + 1e-5)

    if mode == "density":
        return norm_d
    elif mode == "social":
        return 0.5 * norm_d + 0.5 * (1 - order)
    elif mode == "fluid":
        return 0.8 * norm_d
    else:  # proposed
        return 0.4 * norm_d + 0.4 * norm_s + 0.2 * (1 - order)

# -----------------------------
# CROWD DYNAMICS
# -----------------------------
def update_crowd():
    global positions, velocities
    for i in range(N):
        force = np.zeros(2)
        for j in range(N):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)

            if dist < interaction_radius and dist > 1e-3:
                normal = diff / dist
                Gn = normal * (interaction_radius - dist) * 3
                Gt = -0.4 * velocities[i]
                force += Gn + Gt

        velocities[i] += force * dt

        speed = np.linalg.norm(velocities[i])
        if speed > v0:
            velocities[i] = velocities[i] / speed * v0

        positions[i] += velocities[i] * dt
        positions[i] = np.clip(positions[i], 0, L)

# -----------------------------
# A* PATH PLANNING
# -----------------------------
def astar(start, goal, density, disorder):
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq = [(0, start)]
    cost = {start: 0}
    parent = {}

    while pq:
        _, cur = heapq.heappop(pq)

        if cur == goal:
            path = []
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                new_cost = cost[cur] + 1 + 2*density[nx][ny] + 5*disorder[nx][ny]

                if (nx,ny) not in cost or new_cost < cost[(nx,ny)]:
                    cost[(nx,ny)] = new_cost
                    heapq.heappush(pq, (new_cost + h((nx,ny), goal), (nx,ny)))
                    parent[(nx,ny)] = cur
    return None

# -----------------------------
# PATH SAFETY
# -----------------------------
def path_safety(path, density, disorder):
    if not path:
        return 0
    score = 0
    for p in path:
        score += density[p] + disorder[p]
    return score / len(path)

# -----------------------------
# INTERACTION
# -----------------------------
def onclick(event):
    global target_point, start_point
    if event.xdata and event.ydata:
        target_point = (event.xdata, event.ydata)
        corners = [(0,0),(0,L),(L,0),(L,L)]
        start_point = min(corners, key=lambda c: np.linalg.norm(np.array(c)-np.array(target_point)))

# -----------------------------
# VISUALIZATION
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
fig.canvas.mpl_connect('button_press_event', onclick)

def update(frame):
    ax1.clear()
    ax2.clear()

    update_crowd()

    density, disorder = compute_grid()
    order = compute_order()

    risk_map = compute_risk_model(density, disorder, order, "proposed")

    order_history.append(order)
    risk_history.append(np.mean(risk_map))

    # Comparison
    for key in comparison_history:
        r = compute_risk_model(density, disorder, order, key)
        comparison_history[key].append(np.mean(r))

    # -----------------------------
    # PLOT 1: HEATMAP + AGENTS
    # -----------------------------
    ax1.imshow(risk_map.T, origin='lower', extent=[0,L,0,L], cmap='RdYlGn_r', alpha=0.6)

    ax1.scatter(positions[:,0], positions[:,1], s=5, color='black')

    # Path
    if start_point and target_point:
        sx = int(start_point[0]/L*grid_size)
        sy = int(start_point[1]/L*grid_size)
        gx = int(target_point[0]/L*grid_size)
        gy = int(target_point[1]/L*grid_size)

        path = astar((sx,sy),(gx,gy),density,disorder)

        if path:
            px = [p[0]/grid_size*L for p in path]
            py = [p[1]/grid_size*L for p in path]
            ax1.plot(px, py, color='blue', linewidth=2)

            safety = path_safety(path, density, disorder)
            ax1.set_title(f"Risk Map + Path (Safety={safety:.2f})")

    ax1.set_xlim(0,L)
    ax1.set_ylim(0,L)

    # -----------------------------
    # PLOT 2: COMPARISON
    # -----------------------------
    for key in comparison_history:
        ax2.plot(comparison_history[key], label=key)

    ax2.set_title("Model Comparison (Risk)")
    ax2.legend()
    ax2.grid()

ani = FuncAnimation(fig, update, interval=50)
plt.show()

# -----------------------------
# FINAL PLOTS
# -----------------------------
plt.figure()
plt.plot(order_history, label="Order Φ")
plt.plot(risk_history, label="Risk R")
plt.legend()
plt.title("Order vs Risk")
plt.show()

plt.figure()
plt.scatter(order_history, risk_history)
plt.xlabel("Order Φ")
plt.ylabel("Risk R")
plt.title("Phase Transition Plot")
plt.show()