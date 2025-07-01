import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# Load and parse the data
bvh_events = []
gnum_markers = []

with open("profile_report.json") as f:
    for line in f:
        try:
            obj = json.loads(line)
            if "NvtxEvent" in obj:
                evt = obj["NvtxEvent"]
                name = evt.get("Text", "")
                ts = int(evt["Timestamp"])
                if name.startswith("train_") and name.endswith("_bvh"):
                    end_ts = int(evt["EndTimestamp"])
                    dur_ms = (end_ts - ts) / 1e6
                    step = int(name.split("_")[1])
                    bvh_events.append((ts, dur_ms, step))
                elif name.startswith("gnum_"):
                    gnum = int(name.split("_")[1])
                    gnum_markers.append((ts, gnum))
        except:
            continue

# Sort by timestamp to ensure ordering
bvh_events.sort()
gnum_markers.sort()

# Match each BVH event with the nearest previous gnum marker
bvh_vs_particles = []
g_idx = 0

for ts, dur_ms, step in bvh_events:
    while g_idx + 1 < len(gnum_markers) and gnum_markers[g_idx + 1][0] <= ts:
        g_idx += 1
    if g_idx < len(gnum_markers):
        gnum = gnum_markers[g_idx][1]
        bvh_vs_particles.append((gnum, dur_ms))

# Separate values for plotting
particle_counts, bvh_times = zip(*bvh_vs_particles)

# Compute correlation
corr, _ = pearsonr(particle_counts, bvh_times)
print(f"Pearson correlation: {corr:.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(particle_counts, bvh_times, alpha=0.7)
plt.xlabel("Particle Count")
plt.ylabel("BVH Build Time (ms)")
plt.title(f"BVH Build Time vs. Particle Count\n(Pearson r = {corr:.4f})")
plt.grid(True)
plt.tight_layout()
plt.show()
