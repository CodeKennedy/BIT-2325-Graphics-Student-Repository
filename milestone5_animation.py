import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─────────────────────────────────────────────────────
# MILESTONE 5: Dynamics & Animation
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

def generate_heightmap(width=64, height=64, scale=30.0, octaves=5, seed=42):
    hmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            amp=1.0; freq=1.0; val=0.0
            for _ in range(octaves):
                val += perlin((x/scale)*freq,(y/scale)*freq,seed)*amp
                amp*=0.5; freq*=2.0
            hmap[y][x] = val
    return (hmap - hmap.min())/(hmap.max() - hmap.min())

def terrain_colour_array(heightmap):
    H, W = heightmap.shape
    colours = np.zeros((H, W, 3), dtype=np.uint8)
    colours[heightmap <  0.2] = [65,  105, 225]
    colours[(heightmap >= 0.2) & (heightmap < 0.3)] = [100, 149, 237]
    colours[(heightmap >= 0.3) & (heightmap < 0.4)] = [238, 214, 175]
    colours[(heightmap >= 0.4) & (heightmap < 0.6)] = [34,  139,  34]
    colours[(heightmap >= 0.6) & (heightmap < 0.75)]= [85,  107,  47]
    colours[(heightmap >= 0.75)& (heightmap < 0.88)]= [139,  90,  43]
    colours[heightmap >= 0.88] = [240, 240, 240]
    return colours


# ════════════════════════════════════════════════════
# 1. ANIMATED TERRAIN — TIME-SHIFTED PERLIN NOISE
# ════════════════════════════════════════════════════
def generate_animated_heightmap(width=64, height=64,
                                 scale=30.0, octaves=5,
                                 seed=42, time_offset=0.0):
    hmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            amp=1.0; freq=1.0; val=0.0
            for _ in range(octaves):
                nx = ((x / scale) + time_offset) * freq
                ny = (y / scale) * freq
                val += perlin(nx, ny, seed) * amp
                amp*=0.5; freq*=2.0
            hmap[y][x] = val
    return (hmap - hmap.min())/(hmap.max() - hmap.min())


# ════════════════════════════════════════════════════
# 2. PARTICLE SYSTEM — EROSION SIMULATION
# ════════════════════════════════════════════════════
class Particle:
    def __init__(self, x, y):
        self.x = x; self.y = y
        self.vx = np.random.uniform(-0.3, 0.3)
        self.vy = np.random.uniform(-0.3, 0.3)
        self.alive = True
        self.age = 0

class ParticleSystem:
    def __init__(self, heightmap, num_particles=150, seed=7):
        np.random.seed(seed)
        self.heightmap = heightmap
        self.H, self.W = heightmap.shape
        self.particles = [
            Particle(
                x=np.random.uniform(1, self.W-2),
                y=np.random.uniform(1, self.H-2)
            )
            for _ in range(num_particles)
        ]

    def step(self, dt=0.5, gravity=0.1, damping=0.85, max_age=60):
        H, W = self.H, self.W
        for p in self.particles:
            if not p.alive:
                continue
            xi = int(np.clip(p.x, 1, W-2))
            yi = int(np.clip(p.y, 1, H-2))
            dhdx = (self.heightmap[yi][xi+1] - self.heightmap[yi][xi-1]) / 2.0
            dhdy = (self.heightmap[yi+1][xi] - self.heightmap[yi-1][xi]) / 2.0
            p.vx += -dhdx * gravity
            p.vy += -dhdy * gravity
            p.vx *= damping
            p.vy *= damping
            p.x  += p.vx * dt
            p.y  += p.vy * dt
            p.age += 1
            if p.x < 0 or p.x >= W or p.y < 0 or p.y >= H or p.age > max_age:
                p.alive = False

    def respawn_dead(self):
        H, W = self.H, self.W
        for p in self.particles:
            if not p.alive:
                p.x   = np.random.uniform(1, W-2)
                p.y   = np.random.uniform(1, H-2)
                p.vx  = np.random.uniform(-0.3, 0.3)
                p.vy  = np.random.uniform(-0.3, 0.3)
                p.alive = True
                p.age   = 0

    def get_positions(self):
        return [(p.x, p.y) for p in self.particles if p.alive]


# ════════════════════════════════════════════════════
# 3. KEYFRAME CAMERA ORBIT — FIXED
# ════════════════════════════════════════════════════
def keyframe_camera_positions(num_frames):
    """
    Generates exactly num_frames azimuth angles
    by evenly spacing them from 0 to 360 degrees
    with smoothstep interpolation.
    """
    azimuths = []
    for f in range(num_frames):
        t = f / (num_frames - 1)
        t_smooth = fade(t)
        azimuths.append(lerp(0, 360, t_smooth))
    return azimuths


# ════════════════════════════════════════════════════
# 4. GENERATE ANIMATION FRAMES
# ════════════════════════════════════════════════════
print("Generating heightmap...")
NUM_FRAMES  = 30
heightmap   = generate_heightmap(64, 64)
psys        = ParticleSystem(heightmap, num_particles=150)
azimuths    = keyframe_camera_positions(NUM_FRAMES)  # always exactly NUM_FRAMES

frames           = []
stability_errors = []
prev_hmap        = None

print(f"Generating {NUM_FRAMES} animation frames...")
for f in range(NUM_FRAMES):
    t_offset    = f * 0.015
    anim_hmap   = generate_animated_heightmap(64, 64, time_offset=t_offset)
    anim_colour = terrain_colour_array(anim_hmap)

    psys.step()
    psys.respawn_dead()
    positions = psys.get_positions()

    if prev_hmap is not None:
        err = np.mean(np.abs(anim_hmap - prev_hmap))
        stability_errors.append(err)
    prev_hmap = anim_hmap.copy()

    frames.append((anim_colour, positions, azimuths[f]))

print(f"  {NUM_FRAMES} frames generated successfully")
print(f"  Mean frame-to-frame change: {np.mean(stability_errors):.6f}")
print(f"  Max frame-to-frame change:  {np.max(stability_errors):.6f}")


# ════════════════════════════════════════════════════
# 5. STATIC MULTI-FRAME PREVIEW
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Milestone 5 — PCG: Dynamics & Animation\nJames Kennedy | SCT221-C004-0046", fontsize=12)

sample_frames = [0, 4, 8, 12, 16, 20, 24, 29]
for idx, fi in enumerate(sample_frames):
    ax = axes[idx//4][idx%4]
    anim_colour, positions, azimuth = frames[fi]
    ax.imshow(anim_colour)
    if positions:
        px = [p[0] for p in positions]
        py = [p[1] for p in positions]
        ax.scatter(px, py, s=2, c='white', alpha=0.6)
    ax.set_title(f"Frame {fi} | Az={azimuth:.0f}°")
    ax.axis('off')

plt.tight_layout()
plt.savefig("milestone5_output.png", dpi=150)
plt.show()


# ════════════════════════════════════════════════════
# 6. ANIMATED GIF
# ════════════════════════════════════════════════════
print("\nSaving animated GIF...")
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.axis('off')
im = ax2.imshow(frames[0][0])

def update(fi):
    anim_colour, positions, azimuth = frames[fi]
    im.set_data(anim_colour)
    ax2.set_title(f"PCG Animated Terrain — Frame {fi}", fontsize=9)
    return [im]

ani = animation.FuncAnimation(fig2, update, frames=NUM_FRAMES,
                               interval=80, blit=True)
ani.save("milestone5_animated.gif", writer='pillow', fps=12)
plt.close(fig2)

print("Done! Outputs saved:")
print("  milestone5_output.png   — 8-frame static preview")
print("  milestone5_animated.gif — animated terrain loop")

# ════════════════════════════════════════════════════
# 7. MOTION & STABILITY REPORT
# ════════════════════════════════════════════════════
print("\n── Motion & Stability Analysis ──")
print(f"  Total frames:              {NUM_FRAMES}")
print(f"  Particle count:            150")
print(f"  Mean frame delta:          {np.mean(stability_errors):.6f}  (low = smooth)")
print(f"  Max frame delta:           {np.max(stability_errors):.6f}  (low = stable)")
print(f"  Camera orbit:              0° to 360° smoothstep")
print(f"  Integration method:        Euler (dt=0.5, damping=0.85)")