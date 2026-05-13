import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────────────
# MILESTONE 4: Efficiency & Stochastic Methods
# Procedural Content Generation System — James Kennedy
# Reg: SCT221-C004-0046
# ─────────────────────────────────────────────────────

# ── Reused: Vec3 ──
class Vec3:
    def __init__(self, x, y, z):
        self.x = x; self.y = y; self.z = z
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def dot(self, o): return self.x*o.x + self.y*o.y + self.z*o.z
    def length(self): return (self.x**2 + self.y**2 + self.z**2)**0.5
    def normalize(self):
        l = self.length()
        if l == 0: return Vec3(0,0,0)
        return Vec3(self.x/l, self.y/l, self.z/l)


# ── Reused: Perlin Noise ──
def fade(t): return t*t*t*(t*(t*6-15)+10)
def lerp(a, b, t): return a + t*(b-a)
def grad(h, x, y):
    h = h & 3
    if h==0: return  x+y
    if h==1: return -x+y
    if h==2: return  x-y
    return           -x-y

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


# ── Reused: Heightmap & Normals ──
def generate_heightmap(width=128, height=128, scale=40.0, octaves=6, seed=42):
    hmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            amp=1.0; freq=1.0; val=0.0
            for _ in range(octaves):
                val += perlin((x/scale)*freq,(y/scale)*freq,seed)*amp
                amp*=0.5; freq*=2.0
            hmap[y][x] = val
    return (hmap - hmap.min())/(hmap.max() - hmap.min())

def compute_normals(heightmap):
    H, W = heightmap.shape
    normals = np.zeros((H, W, 3))
    for y in range(1, H-1):
        for x in range(1, W-1):
            dhdx = (heightmap[y][x+1] - heightmap[y][x-1]) / 2.0
            dhdz = (heightmap[y+1][x] - heightmap[y-1][x]) / 2.0
            n = Vec3(-dhdx, 1.0, -dhdz).normalize()
            normals[y][x] = [n.x, n.y, n.z]
    return normals

def terrain_colour(h):
    if h < 0.2:  return (65,  105, 225)
    if h < 0.3:  return (100, 149, 237)
    if h < 0.4:  return (238, 214, 175)
    if h < 0.6:  return (34,  139,  34)
    if h < 0.75: return (85,  107,  47)
    if h < 0.88: return (139,  90,  43)
    return               (240, 240, 240)


# ════════════════════════════════════════════════════
# 1. BASELINE RENDER (unoptimised — pure Python loops)
# ════════════════════════════════════════════════════
def phong_slow(normal, light_dir, view_dir):
    N = Vec3(*normal).normalize()
    L = Vec3(*light_dir).normalize()
    V = Vec3(*view_dir).normalize()
    ka=0.2; kd=0.7; ks=0.3; n=32
    NdotL = max(0.0, N.dot(L))
    R = Vec3(2*NdotL*N.x-L.x, 2*NdotL*N.y-L.y, 2*NdotL*N.z-L.z).normalize()
    return min(1.0, ka + kd*NdotL + ks*(max(0.0,R.dot(V))**n))

print("Generating heightmap...")
heightmap = generate_heightmap(128, 128)
normals   = compute_normals(heightmap)
H, W      = heightmap.shape
light_dir = [0.6, 1.0, 0.4]
view_dir  = [0.0, 1.0, 0.5]

print("Running baseline render (slow Python loops)...")
t0 = time.time()
render_baseline = np.zeros((H, W, 3), dtype=np.uint8)
for y in range(H):
    for x in range(W):
        h = heightmap[y][x]
        col = terrain_colour(h)
        if 0 < y < H-1 and 0 < x < W-1:
            intensity = phong_slow(normals[y][x], light_dir, view_dir)
        else:
            intensity = 0.5
        render_baseline[y][x] = [min(255,int(c*intensity)) for c in col]
t_baseline = time.time() - t0
print(f"  Baseline time: {t_baseline:.3f} seconds")


# ════════════════════════════════════════════════════
# 2. OPTIMISED RENDER (vectorised NumPy — no loops)
# ════════════════════════════════════════════════════
print("\nRunning optimised render (NumPy vectorised)...")
t1 = time.time()

L = np.array([0.6, 1.0, 0.4])
L = L / np.linalg.norm(L)
V = np.array([0.0, 1.0, 0.5])
V = V / np.linalg.norm(V)

# Vectorised Phong — operates on entire arrays at once
N = normals / (np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8)
NdotL = np.clip(np.sum(N * L, axis=2), 0, 1)

R = 2 * NdotL[:,:,np.newaxis] * N - L
R = R / (np.linalg.norm(R, axis=2, keepdims=True) + 1e-8)
RdotV = np.clip(np.sum(R * V, axis=2), 0, 1)

intensity_map = np.clip(0.2 + 0.7*NdotL + 0.3*(RdotV**32), 0, 1)

# Build colour map vectorised
colour_map = np.zeros((H, W, 3), dtype=np.uint8)
colour_map[heightmap <  0.2] = [65,  105, 225]
colour_map[(heightmap >= 0.2) & (heightmap < 0.3)] = [100, 149, 237]
colour_map[(heightmap >= 0.3) & (heightmap < 0.4)] = [238, 214, 175]
colour_map[(heightmap >= 0.4) & (heightmap < 0.6)] = [34,  139,  34]
colour_map[(heightmap >= 0.6) & (heightmap < 0.75)]= [85,  107,  47]
colour_map[(heightmap >= 0.75)& (heightmap < 0.88)]= [139,  90,  43]
colour_map[heightmap >= 0.88] = [240, 240, 240]

render_optimised = np.clip(
    colour_map * intensity_map[:,:,np.newaxis], 0, 255
).astype(np.uint8)

t_optimised = time.time() - t1
print(f"  Optimised time: {t_optimised:.3f} seconds")
speedup = t_baseline / t_optimised
print(f"  Speedup: {speedup:.1f}x faster")


# ════════════════════════════════════════════════════
# 3. ACCELERATION STRUCTURE — LEVEL OF DETAIL (LOD)
# ════════════════════════════════════════════════════
def build_lod(heightmap, levels=3):
    """
    Level of Detail (LOD): store the terrain at multiple
    resolutions. Use low-res for distant areas,
    high-res only where needed.
    Returns list of heightmaps from full to coarsest.
    """
    lods = [heightmap]
    current = heightmap
    for _ in range(levels - 1):
        h, w = current.shape
        # Downsample by averaging 2x2 blocks
        small = current[:h//2*2, :w//2*2].reshape(h//2, 2, w//2, 2).mean(axis=(1,3))
        lods.append(small)
        current = small
    return lods

lods = build_lod(heightmap, levels=3)
print(f"\nLOD levels built:")
for i, lod in enumerate(lods):
    print(f"  Level {i}: {lod.shape[0]}x{lod.shape[1]} ({lod.shape[0]*lod.shape[1]:,} pixels)")


# ════════════════════════════════════════════════════
# 4. MONTE CARLO AMBIENT OCCLUSION SAMPLING
# ════════════════════════════════════════════════════
def monte_carlo_ao(heightmap, num_samples=32, radius=8, seed=99):
    """
    Monte Carlo Ambient Occlusion:
    For each point, cast random rays in a hemisphere above
    the surface. Count how many are blocked by terrain.
    Blocked rays = shadow/occlusion. More samples = less noise.
    """
    np.random.seed(seed)
    H, W = heightmap.shape
    ao_map = np.ones((H, W))

    for y in range(1, H-1):
        for x in range(1, W-1):
            h0 = heightmap[y][x]
            blocked = 0
            for _ in range(num_samples):
                # Random direction in hemisphere (uniform sampling)
                theta = np.random.uniform(0, np.pi/2)
                phi   = np.random.uniform(0, 2*np.pi)
                dx = int(np.cos(phi) * np.sin(theta) * radius)
                dy = int(np.sin(phi) * np.sin(theta) * radius)
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if heightmap[ny][nx] > h0 + 0.05:
                        blocked += 1
            ao_map[y][x] = 1.0 - (blocked / num_samples)
    return ao_map

print("\nComputing Monte Carlo Ambient Occlusion (32 samples)...")
t2 = time.time()
ao_map = monte_carlo_ao(heightmap, num_samples=32, radius=8)
t_ao = time.time() - t2
print(f"  AO computation time: {t_ao:.2f} seconds")

# Variance analysis — noise decreases as samples increase
ao_variance = np.var(ao_map)
print(f"  AO map variance (noise level): {ao_variance:.6f}")

# Apply AO to optimised render
render_ao = np.clip(
    render_optimised * ao_map[:,:,np.newaxis], 0, 255
).astype(np.uint8)


# ════════════════════════════════════════════════════
# 5. PERFORMANCE REPORT
# ════════════════════════════════════════════════════
print("\n── Performance Report ──")
print(f"  Baseline (Python loops): {t_baseline:.3f}s")
print(f"  Optimised (NumPy):       {t_optimised:.3f}s")
print(f"  Speedup factor:          {speedup:.1f}x")
print(f"  AO sampling time:        {t_ao:.2f}s")
print(f"  AO noise variance:       {ao_variance:.6f}")


# ════════════════════════════════════════════════════
# 6. VISUALISE
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Milestone 4 — PCG: Efficiency & Stochastic Methods\nJames Kennedy | SCT221-C004-0046", fontsize=12)

axes[0][0].imshow(render_baseline)
axes[0][0].set_title(f"Baseline render ({t_baseline:.2f}s)")
axes[0][0].axis('off')

axes[0][1].imshow(render_optimised)
axes[0][1].set_title(f"Optimised render ({t_optimised:.3f}s) — {speedup:.0f}x faster")
axes[0][1].axis('off')

axes[1][0].imshow(ao_map, cmap='gray')
axes[1][0].set_title("Monte Carlo Ambient Occlusion map")
axes[1][0].axis('off')

axes[1][1].imshow(render_ao)
axes[1][1].set_title("Optimised render + AO applied")
axes[1][1].axis('off')

plt.tight_layout()
plt.savefig("milestone4_output.png", dpi=150)
plt.show()
print("\nDone! Output saved as milestone4_output.png")