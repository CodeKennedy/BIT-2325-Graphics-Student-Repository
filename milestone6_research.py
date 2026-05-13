import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────────────
# MILESTONE 6: Research Contribution
# Procedural Content Generation System — James Kennedy
# Reg: SCT221-C004-0046
#
# NOVEL IDEA: Biome-Aware Adaptive Noise (BAAN)
# Standard PCG applies the same noise parameters
# everywhere. BAAN splits terrain into biome zones
# first, then applies different noise personalities
# per zone — producing richer, more varied landscapes
# from a single generation pass.
# ─────────────────────────────────────────────────────

# ── Reused: Core Math ──
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


# ════════════════════════════════════════════════════
# BASELINE: Standard uniform Perlin terrain
# ════════════════════════════════════════════════════
def standard_heightmap(width=128, height=128,
                        scale=40.0, octaves=6, seed=42):
    hmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            amp=1.0; freq=1.0; val=0.0
            for _ in range(octaves):
                val += perlin((x/scale)*freq,
                              (y/scale)*freq, seed)*amp
                amp*=0.5; freq*=2.0
            hmap[y][x] = val
    return (hmap-hmap.min())/(hmap.max()-hmap.min())


# ════════════════════════════════════════════════════
# NOVEL METHOD: Biome-Aware Adaptive Noise (BAAN)
# ════════════════════════════════════════════════════

# Step 1 — Biome map: low-frequency noise defines zones
def generate_biome_map(width=128, height=128,
                        scale=80.0, seed=99):
    """
    A separate low-frequency noise pass assigns each
    pixel a biome weight between 0 and 1.
    0.0 = desert  |  0.5 = plains  |  1.0 = mountains
    """
    bmap = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            bmap[y][x] = perlin(x/scale, y/scale, seed)
    return (bmap-bmap.min())/(bmap.max()-bmap.min())


# Step 2 — Per-biome noise profiles
BIOME_PROFILES = {
    'desert': {
        'scale': 60.0, 'octaves': 3,
        'persistence': 0.3, 'lacunarity': 2.5,
        'seed': 11
    },
    'plains': {
        'scale': 50.0, 'octaves': 4,
        'persistence': 0.4, 'lacunarity': 2.0,
        'seed': 22
    },
    'mountains': {
        'scale': 25.0, 'octaves': 8,
        'persistence': 0.65, 'lacunarity': 2.2,
        'seed': 33
    }
}

def sample_biome_noise(x, y, profile):
    """Sample fBm noise using a biome-specific profile."""
    amp=1.0
    freq=1.0
    val=0.0
    scale       = profile['scale']
    octaves     = profile['octaves']
    persistence = profile['persistence']
    lacunarity  = profile['lacunarity']
    seed        = profile['seed']
    for _ in range(octaves):
        val += perlin((x/scale)*freq,
                      (y/scale)*freq, seed) * amp
        amp  *= persistence
        freq *= lacunarity
    return val


# Step 3 — BAAN heightmap: blend biomes by weight
def baan_heightmap(width=128, height=128, biome_scale=80.0):
    """
    Biome-Aware Adaptive Noise:
    For each pixel, compute noise from all three biome
    profiles then blend them using the biome map weights.
    Result: smooth transitions between geologically
    distinct terrain regions in one generation pass.
    """
    biome_map = generate_biome_map(width, height, biome_scale)
    hmap = np.zeros((height, width))

    desert    = BIOME_PROFILES['desert']
    plains    = BIOME_PROFILES['plains']
    mountains = BIOME_PROFILES['mountains']

    for y in range(height):
        for x in range(width):
            b = biome_map[y][x]  # 0.0 to 1.0

            # Sample all three biomes
            v_desert    = sample_biome_noise(x, y, desert)
            v_plains    = sample_biome_noise(x, y, plains)
            v_mountains = sample_biome_noise(x, y, mountains)

            # Smooth biome weights using fade curve
            if b < 0.4:
                # Desert zone
                t = b / 0.4
                t = fade(t)
                hmap[y][x] = lerp(v_desert, v_plains, t)
            elif b < 0.7:
                # Plains zone
                t = (b - 0.4) / 0.3
                t = fade(t)
                hmap[y][x] = lerp(v_plains, v_mountains, t)
            else:
                # Mountain zone
                hmap[y][x] = v_mountains

    return (hmap-hmap.min())/(hmap.max()-hmap.min())


# ════════════════════════════════════════════════════
# EVALUATION: Compare standard vs BAAN
# ════════════════════════════════════════════════════
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

def terrain_diversity_score(heightmap, bins=20):
    """
    Measures how evenly height values are distributed.
    Higher entropy = more diverse terrain features.
    Score of 1.0 = perfectly uniform distribution.
    """
    hist, _ = np.histogram(heightmap, bins=bins, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist + 1e-8))
    max_entropy = np.log(bins)
    return entropy / max_entropy

def roughness_score(heightmap):
    """Average absolute gradient — higher = more rugged terrain."""
    dy, dx = np.gradient(heightmap)
    return np.mean(np.sqrt(dx**2 + dy**2))

def biome_transition_smoothness(heightmap):
    """
    Measures second-order smoothness (Laplacian).
    Lower = smoother transitions between regions.
    """
    laplacian = np.abs(np.gradient(np.gradient(heightmap)[0])[0])
    return np.mean(laplacian)


# ════════════════════════════════════════════════════
# RUN COMPARISON
# ════════════════════════════════════════════════════
print("Generating standard terrain...")
t0 = time.time()
hmap_standard = standard_heightmap(128, 128)
t_standard = time.time() - t0

print("Generating BAAN terrain (novel method)...")
t1 = time.time()
hmap_baan = baan_heightmap(128, 128)
t_baan = time.time() - t1

biome_map = generate_biome_map(128, 128)

# Scores
div_std  = terrain_diversity_score(hmap_standard)
div_baan = terrain_diversity_score(hmap_baan)
rug_std  = roughness_score(hmap_standard)
rug_baan = roughness_score(hmap_baan)
smo_std  = biome_transition_smoothness(hmap_standard)
smo_baan = biome_transition_smoothness(hmap_baan)

print("\n── Research Evaluation Report ──")
print(f"{'Metric':<30} {'Standard':>12} {'BAAN':>12}")
print("-" * 56)
print(f"{'Generation time (s)':<30} {t_standard:>12.3f} {t_baan:>12.3f}")
print(f"{'Terrain diversity score':<30} {div_std:>12.4f} {div_baan:>12.4f}")
print(f"{'Roughness score':<30} {rug_std:>12.4f} {rug_baan:>12.4f}")
print(f"{'Transition smoothness':<30} {smo_std:>12.6f} {smo_baan:>12.6f}")
print(f"{'Biome zones':<30} {'1 (uniform)':>12} {'3 (distinct)':>12}")


# ════════════════════════════════════════════════════
# VISUALISE
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Milestone 6 — PCG: Research Contribution\n"
             "Novel Method: Biome-Aware Adaptive Noise (BAAN)\n"
             "James Kennedy | SCT221-C004-0046", fontsize=11)

# Row 1: heightmaps
axes[0][0].imshow(hmap_standard, cmap='terrain')
axes[0][0].set_title("Standard Perlin Heightmap")
axes[0][0].axis('off')

axes[0][1].imshow(biome_map, cmap='RdYlGn')
axes[0][1].set_title("BAAN Biome Map\n(red=desert, green=mountains)")
axes[0][1].axis('off')

axes[0][2].imshow(hmap_baan, cmap='terrain')
axes[0][2].set_title("BAAN Heightmap (Novel Method)")
axes[0][2].axis('off')

# Row 2: coloured terrain
axes[1][0].imshow(terrain_colour_array(hmap_standard))
axes[1][0].set_title(f"Standard Terrain\nDiversity: {div_std:.4f}")
axes[1][0].axis('off')

axes[1][1].imshow(biome_map, cmap='coolwarm')
axes[1][1].set_title("Biome Weight Distribution")
axes[1][1].axis('off')

axes[1][2].imshow(terrain_colour_array(hmap_baan))
axes[1][2].set_title(f"BAAN Terrain\nDiversity: {div_baan:.4f}")
axes[1][2].axis('off')

plt.tight_layout()
plt.savefig("milestone6_output.png", dpi=150)
plt.show()
print("\nDone! Output saved as milestone6_output.png")