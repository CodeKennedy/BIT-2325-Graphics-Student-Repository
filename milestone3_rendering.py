import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────────────────
# MILESTONE 3: Rendering & Signal Processing
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
        if l == 0: return Vec3(0, 0, 0)
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

def generate_heightmap(width=128, height=128, scale=40.0, octaves=6, seed=42):
    hmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            amp=1.0; freq=1.0; val=0.0
            for _ in range(octaves):
                val += perlin((x/scale)*freq, (y/scale)*freq, seed)*amp
                amp*=0.5; freq*=2.0
            hmap[y][x] = val
    return (hmap - hmap.min()) / (hmap.max() - hmap.min())


# ── 1. COMPUTE SURFACE NORMALS ──
def compute_normals(heightmap):
    """
    Estimate surface normal at each point using
    central differences between neighbouring heights.
    Normal = normalize((-dh/dx, 1, -dh/dz))
    """
    H, W = heightmap.shape
    normals = np.zeros((H, W, 3))
    for y in range(1, H-1):
        for x in range(1, W-1):
            dhdx = (heightmap[y][x+1] - heightmap[y][x-1]) / 2.0
            dhdz = (heightmap[y+1][x] - heightmap[y-1][x]) / 2.0
            n = Vec3(-dhdx, 1.0, -dhdz).normalize()
            normals[y][x] = [n.x, n.y, n.z]
    return normals


# ── 2. PHONG SHADING MODEL ──
def phong_shade(normal_arr, light_dir, view_dir, height_val):
    """
    Phong shading: I = ka*Ia + kd*(N.L)*Id + ks*(R.V)^n*Is
    ka: ambient coefficient
    kd: diffuse coefficient
    ks: specular coefficient
    n:  shininess
    """
    N = Vec3(*normal_arr).normalize()
    L = Vec3(*light_dir).normalize()
    V = Vec3(*view_dir).normalize()

    # Ambient
    ka = 0.2
    ambient = ka

    # Diffuse (Lambertian)
    kd = 0.7
    NdotL = max(0.0, N.dot(L))
    diffuse = kd * NdotL

    # Specular (Phong reflection)
    ks = 0.3
    shininess = 32
    R = Vec3(
        2 * NdotL * N.x - L.x,
        2 * NdotL * N.y - L.y,
        2 * NdotL * N.z - L.z
    ).normalize()
    RdotV = max(0.0, R.dot(V))
    specular = ks * (RdotV ** shininess)

    return min(1.0, ambient + diffuse + specular)


# ── 3. TERRAIN COLOUR TEXTURE ──
def terrain_colour(height):
    """
    Assign colour based on height — acts as a procedural texture.
    Returns RGB tuple (0-255).
    """
    if height < 0.2:   return (65,  105, 225)   # deep water — blue
    if height < 0.3:   return (100, 149, 237)   # shallow water
    if height < 0.4:   return (238, 214, 175)   # sand/beach
    if height < 0.6:   return (34,  139,  34)   # lowland grass
    if height < 0.75:  return (85,  107,  47)   # highland grass
    if height < 0.88:  return (139,  90,  43)   # rocky mountain
    return                     (240, 240, 240)   # snow peak


# ── 4. RENDER WITH SHADING ──
print("Rendering terrain with Phong shading...")
heightmap = generate_heightmap(128, 128)
normals   = compute_normals(heightmap)

light_dir = [0.6, 1.0, 0.4]   # sunlight direction
view_dir  = [0.0, 1.0, 0.5]   # viewer direction

H, W = heightmap.shape
render_shaded   = np.zeros((H, W, 3), dtype=np.uint8)
render_flat     = np.zeros((H, W, 3), dtype=np.uint8)

for y in range(H):
    for x in range(W):
        h  = heightmap[y][x]
        col = terrain_colour(h)

        # Flat shading — colour only, no lighting
        render_flat[y][x] = col

        # Phong shading — apply lighting to colour
        if y > 0 and y < H-1 and x > 0 and x < W-1:
            intensity = phong_shade(normals[y][x], light_dir, view_dir, h)
        else:
            intensity = 0.5

        render_shaded[y][x] = [
            min(255, int(col[0] * intensity)),
            min(255, int(col[1] * intensity)),
            min(255, int(col[2] * intensity))
        ]


# ── 5. ANTIALIASING — 2x2 SUPERSAMPLING ──
def supersample(image, factor=2):
    """
    Supersampling antialiasing: render at higher resolution
    then downsample by averaging factor x factor pixel blocks.
    Reduces aliasing (jagged edges) at boundaries.
    """
    h, w = image.shape[:2]
    small_h, small_w = h // factor, w // factor
    result = np.zeros((small_h, small_w, 3), dtype=np.uint8)
    for y in range(small_h):
        for x in range(small_w):
            block = image[y*factor:(y+1)*factor, x*factor:(x+1)*factor]
            result[y][x] = block.mean(axis=(0,1)).astype(np.uint8)
    return result

# Render at 2x size then downsample for antialiasing comparison
print("Applying supersampling antialiasing...")
render_aa = supersample(render_shaded, factor=2)


# ── 6. ARTEFACT ANALYSIS ──
# Compare aliased vs antialiased using pixel variance along an edge row
row_aliased = render_shaded[64, :, 0].astype(float)
row_aa      = render_aa[32, :, 0].astype(float)

variance_before = np.var(np.diff(row_aliased))
variance_after  = np.var(np.diff(row_aa))

print(f"\nAliasing artefact analysis:")
print(f"  Pixel variance before AA: {variance_before:.2f}")
print(f"  Pixel variance after AA:  {variance_after:.2f}")
print(f"  Reduction: {((variance_before - variance_after)/variance_before*100):.1f}%")


# ── 7. VISUALISE ALL OUTPUTS ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Milestone 3 — PCG: Rendering & Signal Processing\nJames Kennedy | SCT221-C004-0046", fontsize=12)

axes[0].imshow(render_flat)
axes[0].set_title("Strategy 1: Flat shading (colour only)")
axes[0].axis('off')

axes[1].imshow(render_shaded)
axes[1].set_title("Strategy 2: Phong shading (with lighting)")
axes[1].axis('off')

axes[2].imshow(render_aa)
axes[2].set_title("Strategy 3: Phong + Supersampling AA")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("milestone3_output.png", dpi=150)
plt.show()

# Save individual renders
Image.fromarray(render_flat).save("milestone3_flat.png")
Image.fromarray(render_shaded).save("milestone3_phong.png")
Image.fromarray(render_aa).save("milestone3_antialiased.png")

print("\nDone! Outputs saved.")
