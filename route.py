import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
CONGESTION_LEVEL = 2 
if CONGESTION_LEVEL == 1:
    N = 250; interaction_radius = 0.8
elif CONGESTION_LEVEL == 2:
    N = 400; interaction_radius = 0.7
else:
    N = 550; interaction_radius = 0.6
L = 10
v0 = 0.25
dt = 0.05
grid_size = 40
sigma = np.deg2rad(15)
positions = np.random.rand(N, 2) * L
angles = np.random.uniform(-np.pi, np.pi, N)
velocities = np.column_stack((np.cos(angles), np.sin(angles))) * v0
start_point = None
target_point = None
agent_pos = None
agent_path = []
path_index = 0
frame_count = 0
recompute_interval = 10
order_history = []
density_history = []
disorder_history = []
def compute_order():
    return np.linalg.norm(np.mean(velocities, axis=0)) / v0
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
            velocities[i] = velocities[i]/speed * v0
        positions[i] += velocities[i] * dt
        positions[i] = np.clip(positions[i], 0, L)
def compute_grid():
    density = np.zeros((grid_size, grid_size))
    disorder = np.zeros((grid_size, grid_size))
    angle_map = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    for p, v in zip(positions, velocities):
        x = int(p[0]/L * grid_size)
        y = int(p[1]/L * grid_size)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            density[x][y] += 1
            angle_map[x][y].append(np.arctan2(v[1], v[0]))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(angle_map[i][j]) > 1:
                disorder[i][j] = np.std(angle_map[i][j])
    return density, disorder
def astar_k_paths(start, goal, density, disorder, K=3):
    def h(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    def astar(penalty):
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
                return path[::-1], cost[cur]
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = cur[0]+dx, cur[1]+dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    d = density[nx][ny]
                    s = disorder[nx][ny]
                    p = penalty.get((nx,ny),0)
                    new_cost = cost[cur] + 1 + 2*d + 4*s + p
                    if (nx,ny) not in cost or new_cost < cost[(nx,ny)]:
                        cost[(nx,ny)] = new_cost
                        heapq.heappush(pq,(new_cost + h((nx,ny),goal),(nx,ny)))
                        parent[(nx,ny)] = cur
        return None, float('inf')
    paths = []
    penalty = {}
    for _ in range(K):
        path, c = astar(penalty)
        if not path:
            break
        paths.append((path, c))
        for p in path:
            penalty[p] = penalty.get(p,0) + 5
    return paths
def move_agent():
    global agent_pos, path_index
    if agent_path and path_index < len(agent_path):
        target = agent_path[path_index]
        dx = target[0] - agent_pos[0]
        dy = target[1] - agent_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.1:
            path_index += 1
        else:
            agent_pos = (
                agent_pos[0] + 0.2 * dx,
                agent_pos[1] + 0.2 * dy
            )
def onclick(event):
    global start_point, target_point, agent_pos
    if event.xdata is not None and event.ydata is not None:
        if start_point is None:
            start_point = (event.xdata, event.ydata)
            agent_pos = start_point
            print("Start selected:", start_point)
        elif target_point is None:
            target_point = (event.xdata, event.ydata)
            print("Victim selected:", target_point)
        else:
            target_point = (event.xdata, event.ydata)
            print("Victim moved:", target_point)
def onkey(event):
    global start_point, target_point, agent_pos, agent_path, path_index
    if event.key == 'r':
        start_point = None
        target_point = None
        agent_pos = None
        agent_path = []
        path_index = 0
        print("Reset done")
def update(frame):
    global frame_count, agent_path, path_index
    frame_count += 1
    update_crowd()
    ax.clear()
    ax.scatter(positions[:,0], positions[:,1], s=5, color='black')
    order_history.append(compute_order())
    if start_point:
        ax.scatter(start_point[0], start_point[1], color='blue', s=80)
    if target_point:
        ax.scatter(target_point[0], target_point[1], color='red', s=100)
    if agent_pos:
        ax.scatter(agent_pos[0], agent_pos[1], color='green', s=100)
    if agent_pos and target_point:
        density, disorder = compute_grid()
        density_history.append(np.mean(density))
        disorder_history.append(np.mean(disorder))
        sx = int(agent_pos[0]/L * grid_size)
        sy = int(agent_pos[1]/L * grid_size)
        gx = int(target_point[0]/L * grid_size)
        gy = int(target_point[1]/L * grid_size)
        if frame_count % recompute_interval == 0:
            paths = astar_k_paths((sx,sy),(gx,gy),density,disorder,3)
            if paths:
                paths.sort(key=lambda x: x[1])
                best_path = paths[0][0]
                agent_path = [(p[0]/grid_size * L, p[1]/grid_size * L) for p in best_path]
                path_index = 0
        if agent_path:
            px, py = zip(*agent_path)
            ax.plot(px, py, color='brown', linewidth=3)
            for i in range(0, len(px)-1, 10):
                ax.arrow(px[i], py[i],
                         px[i+1]-px[i], py[i+1]-py[i],
                         head_width=0.15, color='brown')
        move_agent()
    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    ax.set_title("Real-Time Rescue System (Dynamic Switching)")
fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)
ani = FuncAnimation(fig, update, interval=50)
plt.show()
plt.figure(figsize=(10,6))
plt.plot(order_history, label="Order")
plt.plot(density_history, label="Density")
plt.plot(disorder_history, label="Disorder")
plt.legend()
plt.title("Crowd Physics Report")
plt.xlabel("Time")
plt.ylabel("Values")

plt.savefig("crowd_report.png")
plt.show()