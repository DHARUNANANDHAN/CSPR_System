import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Simulation Parameters
# -----------------------------
L = float(input("Enter area size (meters): "))
N = int(input("Enter number of people: "))

density_value = N / (L * L)
if density_value < 0.01:
    print("Too sparse → Simulation may look empty")
elif density_value > 5:
    print("Too dense → Extreme congestion possible")

v0 = 0.3               # preferred speed
dt = 0.05              # time step
interaction_radius = 0.7
grid_size = int(L * 3) # finer grid for heatmaps

# Initialize positions and velocities
positions = np.random.rand(N, 2) * L
angles = np.random.uniform(-np.pi, np.pi, N)
velocities = np.column_stack((np.cos(angles), np.sin(angles))) * v0

# Store history
order_history = []
risk_history = []

# -----------------------------
# Crowd Parameters Functions
# -----------------------------
def compute_order():
    """Order parameter Φ"""
    mean_vel = np.mean(velocities, axis=0)
    return np.linalg.norm(mean_vel) / v0

def compute_grid():
    """Compute density and directional disorder per cell"""
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

def compute_risk(density, disorder, order):
    """Compute combined risk function R = 0.4ρ + 0.4σ + 0.2(1-Φ)"""
    # Normalize density and disorder to [0,1]
    norm_density = density / (np.max(density)+1e-5)
    norm_disorder = disorder / (np.max(disorder)+1e-5)
    return 0.4 * norm_density + 0.4 * norm_disorder + 0.2 * (1 - order)

# -----------------------------
# Update Crowd Dynamics
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
                Gn = normal * (interaction_radius - dist) * 2
                Gt = -0.3 * velocities[i]
                force += Gn + Gt
        velocities[i] += force * dt
        # Limit speed to v0
        speed = np.linalg.norm(velocities[i])
        if speed > v0:
            velocities[i] = velocities[i] / speed * v0
        positions[i] += velocities[i] * dt
        # Keep inside bounds
        positions[i] = np.clip(positions[i], 0, L)

# -----------------------------
# Visualization Setup
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def update(frame):
    ax1.clear()
    ax2.clear()

    # Update positions
    update_crowd()

    # Compute crowd parameters
    density, disorder = compute_grid()
    order = compute_order()
    risk_map = compute_risk(density, disorder, order)

    # Store history
    order_history.append(order)
    risk_history.append(np.mean(risk_map))

    # -----------------------------
    # Risk Heatmap + Agents
    # -----------------------------
    ax1.imshow(risk_map.T,
               origin='lower',
               extent=[0, L, 0, L],
               cmap='RdYlGn_r',
               alpha=0.6)
    # Velocity vectors
    skip = max(1, N // 100)  # skip vectors if too many agents
    ax1.quiver(positions[::skip,0], positions[::skip,1],
               velocities[::skip,0], velocities[::skip,1],
               color='black', scale=5, width=0.002)
    # Agent positions
    ax1.scatter(positions[:,0], positions[:,1], s=5, color='blue')

    ax1.set_title(f"Stampede Risk Map and Velocity Vectors")
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)

    # -----------------------------
    # Order–Disorder Plot
    # -----------------------------
    ax2.plot(order_history, label="Order Φ", color='green')
    ax2.plot(risk_history, label="Mean Risk R", color='red')
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Value")
    ax2.set_title("Order–Disorder Transition Over Time")
    ax2.legend()
    ax2.grid(True)

# -----------------------------
# Run Animation
# -----------------------------
ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()