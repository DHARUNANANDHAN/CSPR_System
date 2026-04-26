import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# -----------------------------
# Simulation Parameters
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
grid_size = int(L*4)  # finer grid for heatmaps

# Initialize positions and velocities
positions = np.random.rand(N, 2) * L
angles = np.random.uniform(-np.pi, np.pi, N)
velocities = np.column_stack((np.cos(angles), np.sin(angles))) * v0

# Target and start points for adaptive path planning
target_point = None
start_point = None

# -----------------------------
# History Storage
# -----------------------------
order_history = []
risk_history = []

# -----------------------------
# Crowd Parameter Functions
# -----------------------------
def compute_order():
    """Order parameter Φ"""
    return np.linalg.norm(np.mean(velocities, axis=0)) / v0

def compute_grid():
    """Compute density and directional disorder"""
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
# Crowd Update Dynamics
# -----------------------------
def update_crowd():
    global positions, velocities
    for i in range(N):
        force = np.zeros(2)
        for j in range(N):
            if i == j: continue
            diff = positions[i]-positions[j]
            dist = np.linalg.norm(diff)
            if dist < interaction_radius and dist > 1e-3:
                normal = diff/dist
                Gn = normal*(interaction_radius-dist)*3
                Gt = -0.4*velocities[i]
                force += Gn + Gt
        velocities[i] += force*dt
        # Limit speed
        speed = np.linalg.norm(velocities[i])
        if speed>v0:
            velocities[i] = velocities[i]/speed*v0
        positions[i] += velocities[i]*dt
        positions[i] = np.clip(positions[i],0,L)

# -----------------------------
# Adaptive Path Planning
# -----------------------------
def astar_k_paths(start, goal, density, disorder, K=3):
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    def astar(penalty):
        pq = [(0,start)]
        cost = {start:0}
        parent = {}
        while pq:
            _, cur = heapq.heappop(pq)
            if cur==goal:
                path=[]
                while cur in parent:
                    path.append(cur)
                    cur=parent[cur]
                return path[::-1], cost[cur]
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = cur[0]+dx, cur[1]+dy
                if 0<=nx<grid_size and 0<=ny<grid_size:
                    d=density[nx][ny]
                    s=disorder[nx][ny]
                    p=penalty.get((nx,ny),0)
                    new_cost = cost[cur]+1 + 2*d + 4*s + p
                    if (nx,ny) not in cost or new_cost<cost[(nx,ny)]:
                        cost[(nx,ny)] = new_cost
                        heapq.heappush(pq,(new_cost+h((nx,ny),goal),(nx,ny)))
                        parent[(nx,ny)] = cur
        return None,float('inf')
    paths=[]
    penalty={}
    for _ in range(K):
        path,c=astar(penalty)
        if not path: break
        paths.append((path,c))
        for p in path:
            penalty[p]=penalty.get(p,0)+5
    return paths

# -----------------------------
# Interactive Selection
# -----------------------------
def onclick(event):
    global target_point, start_point
    if event.xdata is not None and event.ydata is not None:
        target_point=(event.xdata,event.ydata)
        corners=[(0,0),(0,L),(L,0),(L,L)]
        start_point=min(corners,key=lambda c: np.linalg.norm(np.array(c)-np.array(target_point)))
        print("Victim:",target_point,"Rescuer (corner):",start_point)

def onkey(event):
    global target_point, start_point
    if event.key=='r':
        target_point=None
        start_point=None
        print("Reset complete")

# -----------------------------
# Visualization
# -----------------------------
fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)

def update(frame):
    update_crowd()
    ax.clear()

    # Plot agents
    ax.scatter(positions[:,0], positions[:,1], s=5, color='black')

    # Plot corners and selected points
    corners=[(0,0),(0,L),(0,L),(L,L)]
    for c in corners: ax.scatter(c[0],c[1],color='blue',s=60)
    if start_point: ax.scatter(start_point[0],start_point[1],color='green',s=100)
    if target_point: ax.scatter(target_point[0],target_point[1],color='red',s=120)

    # Order-disorder tracking
    order_history.append(compute_order())

    # Path planning
    if start_point and target_point:
        density, disorder = compute_grid()
        sx=int(start_point[0]/L*grid_size)
        sy=int(start_point[1]/L*grid_size)
        gx=int(target_point[0]/L*grid_size)
        gy=int(target_point[1]/L*grid_size)
        paths=astar_k_paths((sx,sy),(gx,gy),density,disorder,3)
        if paths:
            paths.sort(key=lambda x:x[1])
            for idx,(path,_) in enumerate(paths):
                px=[p[0]/grid_size*L for p in path]
                py=[p[1]/grid_size*L for p in path]
                if idx==0:
                    ax.plot(px,py,color='brown',linewidth=3)
                    for i in range(0,len(px)-1,10):
                        ax.arrow(px[i],py[i],px[i+1]-px[i],py[i+1]-py[i],head_width=0.15,color='brown')
                else:
                    ax.plot(px,py,linestyle='--',alpha=0.5)

    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    ax.set_title("Crowd Simulation with Adaptive Path Planning")

ani=FuncAnimation(fig,update,interval=50)
plt.show()

# -----------------------------
# Save Order Report
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(order_history,label="Order Φ")
plt.title("Order Parameter Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Φ")
plt.legend()
plt.savefig("order_history.png")
plt.show()