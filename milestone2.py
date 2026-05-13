import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────────────
# MILESTONE 2: Space, Transformation & Camera
# Procedural Content Generation System — James Kennedy
# Reg: SCT221-C004-0046
# ─────────────────────────────────────────────────────

# ── Reuse Vec3 and Mat4 from Milestone 1 ──
class Vec3:
    def __init__(self, x, y, z):
        self.x = x; self.y = y; self.z = z
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def dot(self, o): return self.x*o.x + self.y*o.y + self.z*o.z
    def cross(self, o):
        return Vec3(self.y*o.z - self.z*o.y,
                    self.z*o.x - self.x*o.z,
                    self.x*o.y - self.y*o.x)
    def length(self): return (self.x**2 + self.y**2 + self.z**2)**0.5
    def normalize(self):
        l = self.length()
        return Vec3(self.x/l, self.y/l, self.z/l)
    def to_array(self): return np.array([self.x, self.y, self.z, 1.0])
    def __repr__(self): return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class Mat4:
    def __init__(self, data=None):
        self.m = np.identity(4) if data is None else np.array(data, dtype=float)
    def __matmul__(self, o):
        r = Mat4(); r.m = self.m @ o.m; return r
    def transform_point(self, v):
        p = self.m @ v.to_array()
        return Vec3(p[0], p[1], p[2])

    @staticmethod
    def translation(tx, ty, tz):
        m = Mat4()
        m.m[0][3]=tx; m.m[1][3]=ty; m.m[2][3]=tz
        return m

    @staticmethod
    def scale(sx, sy, sz):
        m = Mat4()
        m.m[0][0]=sx; m.m[1][1]=sy; m.m[2][2]=sz
        return m

    @staticmethod
    def rotation_y(angle_deg):
        a = np.radians(angle_deg)
        m = Mat4()
        m.m[0][0] =  np.cos(a); m.m[0][2] = np.sin(a)
        m.m[2][0] = -np.sin(a); m.m[2][2] = np.cos(a)
        return m

    @staticmethod
    def rotation_x(angle_deg):
        a = np.radians(angle_deg)
        m = Mat4()
        m.m[1][1] =  np.cos(a); m.m[1][2] = -np.sin(a)
        m.m[2][1] =  np.sin(a); m.m[2][2] =  np.cos(a)
        return m


# ── 1. VIEW MATRIX (camera look-at) ──
def look_at(eye, target, up):
    """
    Builds a view matrix from camera position (eye),
    target point, and world up vector.
    """
    f = (target - eye).normalize()       # forward vector
    r = f.cross(up).normalize()          # right vector
    u = r.cross(f)                        # true up vector

    m = np.identity(4)
    m[0][:3] = [r.x, r.y, r.z]
    m[1][:3] = [u.x, u.y, u.z]
    m[2][:3] = [-f.x, -f.y, -f.z]
    m[0][3]  = -r.dot(eye)
    m[1][3]  = -u.dot(eye)
    m[2][3]  =  f.dot(eye)

    view = Mat4()
    view.m = m
    return view


# ── 2. PROJECTION MATRIX (perspective) ──
def perspective(fov_deg, aspect, near, far):
    """
    Builds a perspective projection matrix.
    fov_deg: vertical field of view in degrees
    aspect:  width / height
    near:    near clipping plane
    far:     far clipping plane
    """
    fov = np.radians(fov_deg)
    f = 1.0 / np.tan(fov / 2.0)
    m = np.zeros((4, 4))
    m[0][0] = f / aspect
    m[1][1] = f
    m[2][2] = (far + near) / (near - far)
    m[2][3] = (2 * far * near) / (near - far)
    m[3][2] = -1.0

    proj = Mat4()
    proj.m = m
    return proj


# ── 3. PERLIN NOISE (reused from M1) ──
def fade(t): return t*t*t*(t*(t*6-15)+10)
def lerp(a, b, t): return a + t*(b-a)
def grad(h, x, y):
    h = h & 3
    if h==0: return  x+y
    if h==1: return -x+y
    if h==2: return  x-y
    return            -x-y

def perlin(x, y, seed=0):
    np.random.seed(seed)
    perm = np.arange(256, dtype=int)
    np.random.shuffle(perm)
    perm = np.tile(perm, 2)
    xi=int(x)&255; yi=int(y)&255
    xf=x-int(x);   yf=y-int(y)
    u=fade(xf);     v=fade(yf)
    aa=perm[perm[xi]+yi];   ab=perm[perm[xi]+yi+1]
    ba=perm[perm[xi+1]+yi]; bb=perm[perm[xi+1]+yi+1]
    x1=lerp(grad(aa,xf,yf),    grad(ba,xf-1,yf),   u)
    x2=lerp(grad(ab,xf,yf-1),  grad(bb,xf-1,yf-1), u)
    return lerp(x1, x2, v)

def generate_heightmap(width=64, height=64, scale=30.0, octaves=6, seed=42):
    hmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            amp=1.0; freq=1.0; val=0.0
            for _ in range(octaves):
                val += perlin((x/scale)*freq, (y/scale)*freq, seed)*amp
                amp*=0.5; freq*=2.0
            hmap[y][x] = val
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    return hmap


# ── 4. BUILD 3D TERRAIN MESH ──
print("Building 3D terrain mesh...")
W, H = 64, 64
heightmap = generate_heightmap(W, H)

# Create X, Z grid and Y from heightmap
X = np.linspace(-1, 1, W)
Z = np.linspace(-1, 1, H)
X, Z = np.meshgrid(X, Z)
Y = heightmap * 0.5  # scale height


# ── 5. APPLY TRANSFORMATION PIPELINE ──
# Demonstrate: scale → rotate → translate on a sample point
v = Vec3(0.5, 0.3, 0.5)

S  = Mat4.scale(1.5, 1.5, 1.5)
Ry = Mat4.rotation_y(45)
Rx = Mat4.rotation_x(15)
T  = Mat4.translation(0.1, 0.0, -0.5)

# Combined transform: T * Rx * Ry * S
transform = T @ Rx @ Ry @ S
v_transformed = transform.transform_point(v)

print(f"Original point:    {v}")
print(f"After transform:   {v_transformed}")


# ── 6. CAMERA SETUP ──
eye    = Vec3(2.0,  1.5,  2.0)
target = Vec3(0.0,  0.1,  0.0)
up     = Vec3(0.0,  1.0,  0.0)

view_matrix = look_at(eye, target, up)
proj_matrix = perspective(fov_deg=60, aspect=16/9, near=0.1, far=100.0)

print(f"\nView Matrix:\n{view_matrix.m.round(3)}")
print(f"\nProjection Matrix:\n{proj_matrix.m.round(3)}")

# Numerical stability check
det = np.linalg.det(view_matrix.m)
print(f"\nView matrix determinant: {det:.6f}  (should be ~1.0 for stable transform)")


# ── 7. VISUALISE ──
fig = plt.figure(figsize=(14, 5))
fig.suptitle("Milestone 2 — PCG: Space, Transformation & Camera\nJames Kennedy | SCT221-C004-0046", fontsize=12)

# Plot 1: top-down heightmap
ax1 = fig.add_subplot(131)
ax1.imshow(heightmap, cmap='terrain')
ax1.set_title("Top-down heightmap")
ax1.axis('off')

# Plot 2: 3D terrain from default angle
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Z, Y, cmap='terrain', linewidth=0, antialiased=True)
ax2.set_title("3D terrain (default view)")
ax2.set_xlabel("X"); ax2.set_ylabel("Z"); ax2.set_zlabel("Y")

# Plot 3: 3D terrain from camera angle (simulated)
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Z, Y, cmap='terrain', linewidth=0, antialiased=True)
ax3.view_init(elev=30, azim=135)
ax3.set_title("3D terrain (camera view)")
ax3.set_xlabel("X"); ax3.set_ylabel("Z"); ax3.set_zlabel("Y")

plt.tight_layout()
plt.savefig("milestone2_output.png", dpi=150)
plt.show()
print("\nDone! Output saved as milestone2_output.png")