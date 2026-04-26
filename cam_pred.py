import cv2
import numpy as np
import heapq
import threading
import matplotlib.pyplot as plt

url = "http://192.168.43.192"

class CamThread:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            ret, frame = self.cap.read()
            if ret: 
                with self.lock:
                    self.frame = frame.copy()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

cam = CamThread(url)

grid_size = 40
target_person = None
prev_points = None
velocities = {}
order_history = []

def detect_people(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,th = cv2.threshold(blur,180,255,cv2.THRESH_BINARY_INV)

    contours,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pts=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if 300 < w*h < 5000:
            pts.append((x+w//2,y+h//2))
    return pts
def compute_velocity(points):
    global prev_points, velocities

    if prev_points is None:
        prev_points = points
        return

    new_vel = {}
    for p in points:
        nearest = min(prev_points, key=lambda q: np.linalg.norm(np.array(p)-np.array(q)))
        new_vel[p] = np.array(p) - np.array(nearest)

    velocities = new_vel
    prev_points = points

def compute_grid(points, velocities, W, H):

    density = np.zeros((grid_size, grid_size))
    disorder = np.zeros((grid_size, grid_size))
    angle_map = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

    for p in points:
        x = int(p[0]/W * grid_size)
        y = int(p[1]/H * grid_size)

        if 0 <= x < grid_size and 0 <= y < grid_size:
            density[x][y] += 1

            if p in velocities:
                v = velocities[p]
                angle = np.arctan2(v[1], v[0])
                angle_map[x][y].append(angle)

    for i in range(grid_size):
        for j in range(grid_size):
            if len(angle_map[i][j]) > 1:
                disorder[i][j] = np.std(angle_map[i][j])

    return density, disorder

def compute_order():
    if len(velocities)==0:
        return 0
    vel_array = np.array(list(velocities.values()))
    mean_vel = np.mean(vel_array,axis=0)
    return np.linalg.norm(mean_vel)/(np.mean(np.linalg.norm(vel_array,axis=1))+1e-5)

def astar_k(start, goal, density, disorder, K=3):

    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

    def astar(penalty):
        pq=[(0,start)]
        cost={start:0}
        parent={}

        while pq:
            _,cur=heapq.heappop(pq)

            if cur==goal:
                path=[]
                while cur in parent:
                    path.append(cur)
                    cur=parent[cur]
                return path[::-1],cost[cur]

            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx,ny=cur[0]+dx,cur[1]+dy

                if 0<=nx<grid_size and 0<=ny<grid_size:
                    d = density[nx][ny]
                    s = disorder[nx][ny]
                    p = penalty.get((nx,ny),0)

                    new = cost[cur] + 1 + 2*d + 4*s + p

                    if (nx,ny) not in cost or new < cost[(nx,ny)]:
                        cost[(nx,ny)] = new
                        heapq.heappush(pq,(new+h((nx,ny),goal),(nx,ny)))
                        parent[(nx,ny)] = cur

        return None,float('inf')

    paths=[]
    penalty={}

    for _ in range(K):
        path,c = astar(penalty)
        if not path: break
        paths.append((path,c))

        for p in path:
            penalty[p] = penalty.get(p,0) + 5

    return paths

def click(event,x,y,flags,param):
    global target_person

    if event == cv2.EVENT_LBUTTONDOWN and len(param)>0:
        target_person = min(param, key=lambda p: np.linalg.norm(np.array(p)-np.array((x,y))))

cv2.namedWindow("Hybrid System")

while True:

    frame = cam.read()
    if frame is None:
        continue

    H,W,_ = frame.shape

    points = detect_people(frame)
    compute_velocity(points)

    cv2.setMouseCallback("Hybrid System", click, points)


    for p in points:
        cv2.circle(frame,p,4,(0,0,0),-1)

    start = (W//2, H//2)
    cv2.circle(frame,start,6,(255,0,0),-1)

    if target_person:
        cv2.circle(frame,target_person,8,(0,0,255),-1)

        density, disorder = compute_grid(points, velocities, W, H)
        order = compute_order()
        order_history.append(order)

        sx = int(start[0]/W * grid_size)
        sy = int(start[1]/H * grid_size)
        gx = int(target_person[0]/W * grid_size)
        gy = int(target_person[1]/H * grid_size)

        paths = astar_k((sx,sy),(gx,gy),density,disorder,3)

        if paths:
            paths.sort(key=lambda x:x[1])

            for idx,(path,_) in enumerate(paths):
                pts=[(int(p[0]/grid_size*W),int(p[1]/grid_size*H)) for p in path]

                if idx == 0:
                    for i in range(len(pts)-1):
                        cv2.arrowedLine(frame,pts[i],pts[i+1],(42,42,165),2)
                else:
                    for i in range(len(pts)-1):
                        cv2.line(frame,pts[i],pts[i+1],(0,255,255),1)

        cv2.putText(frame,f"Order: {order:.2f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.imshow("Hybrid System",frame)

    if cv2.waitKey(1)==27:
        break

cv2.destroyAllWindows()

plt.plot(order_history)
plt.title("Order-Disorder Transition")
plt.show()