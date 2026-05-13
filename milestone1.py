import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# MILESTONE 1: Representation & Foundations
# Procedural Content Generation System
# ─────────────────────────────────────────

# ── 1. VECTOR CLASS (built from scratch) ──
class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalize(self):
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l)

    def __repr__(self):
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


# ── 2. MATRIX 4x4 CLASS (built from scratch) ──
class Mat4:
    def __init__(self, data=None):
        if data is None:
            self.m = np.identity(4)
        else:
            self.m = np.array(data, dtype=float)

    def __matmul__(self, other):
        result = Mat4()
        result.m = self.m @ other.m
        return result

    def transform_point(self, v):
        p = np.array([v.x, v.y, v.z, 1.0])
        r = self.m @ p
        return Vec3(r[0], r[1], r[2])

    @staticmethod
    def translation(tx, ty, tz):
        m = Mat4()
        m.m[0][3] = tx
        m.m[1][3] = ty
        m.m[2][3] = tz
        return m

    @staticmethod
    def scale(sx, sy, sz):
        m = Mat4()
        m.m[0][0] = sx
        m.m[1][1] = sy
        m.m[2][2] = sz
        return m

    def __repr__(self):
        return f"Mat4(\n{self.m}\n)"


# ── 3. PERLIN NOISE (implemented from scratch) ──
def fade(t):
    # Smooth interpolation curve: 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b - a)

def grad(hash_val, x, y):
    # Map hash to one of 4 gradient directions
    h = hash_val & 3
    if h == 0: return  x + y
    if h == 1: return -x + y
    if h == 2: return  x - y
    return              -x - y

def perlin(x, y, seed=0):
    np.random.seed(seed)
    perm = np.arange(256, dtype=int)
    np.random.shuffle(perm)
    perm = np.tile(perm, 2)  # duplicate for overflow safety

    xi = int(x) & 255
    yi = int(y) & 255
    xf = x - int(x)
    yf = y - int(y)

    u = fade(xf)
    v = fade(yf)

    aa = perm[perm[xi]     + yi]
    ab = perm[perm[xi]     + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = lerp(grad(aa, xf,     yf),     grad(ba, xf - 1, yf),     u)
    x2 = lerp(grad(ab, xf,     yf - 1), grad(bb, xf - 1, yf - 1), u)
    return lerp(x1, x2, v)


# ── 4. GENERATE TERRAIN HEIGHTMAP ──
def generate_heightmap(width=256, height=256, scale=50.0, octaves=6, seed=42):
    heightmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            amplitude = 1.0
            frequency = 1.0
            value = 0.0
            for _ in range(octaves):
                nx = (x / scale) * frequency
                ny = (y / scale) * frequency
                value += perlin(nx, ny, seed) * amplitude
                amplitude *= 0.5
                frequency *= 2.0
            heightmap[y][x] = value
    # Normalise to 0–1
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    return heightmap


# ── 5. VISUALISE & SAVE OUTPUT ──
print("Generating terrain heightmap...")
heightmap = generate_heightmap(width=256, height=256, scale=50.0, octaves=6)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Milestone 1 — PCG Terrain: Representation & Foundations", fontsize=13)

# Greyscale heightmap
axes[0].imshow(heightmap, cmap='gray')
axes[0].set_title("Heightmap (Perlin Noise)")
axes[0].axis('off')

# Colour terrain map
terrain_colors = plt.cm.terrain(heightmap)
axes[1].imshow(terrain_colors)
axes[1].set_title("Terrain Colour Map")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("milestone1_output.png", dpi=150)
plt.show()

print("Done! Output saved as milestone1_output.png")

# ── 6. MATH VERIFICATION ──
print("\n── Vec3 & Mat4 Verification ──")
v1 = Vec3(1, 2, 3)
v2 = Vec3(4, 5, 6)
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 dot v2 = {v1.dot(v2)}")
print(f"v1 normalised = {v1.normalize()}")

T = Mat4.translation(1, 2, 3)
S = Mat4.scale(2, 2, 2)
TS = T @ S
print(f"\nTranslation matrix applied to v1: {T.transform_point(v1)}")
print(f"Scale matrix applied to v1: {S.transform_point(v1)}")