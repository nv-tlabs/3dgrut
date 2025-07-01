import json
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr
import numpy as np

# Step 1: Load and parse data
bvh_events = []
gnum_markers = []

with open("profile_report_flowers.json") as f:
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
                    bvh_events.append((ts, dur_ms))
                elif name.startswith("gnum_"):
                    gnum = int(name.split("_")[1])
                    gnum_markers.append((ts, gnum))
        except:
            continue

# Step 2: Match BVH events to nearest previous gnum marker
bvh_vs_gnum = []
g_idx = 0
gnum_markers.sort()
bvh_events.sort()

for ts, dur_ms in bvh_events:
    while g_idx + 1 < len(gnum_markers) and gnum_markers[g_idx + 1][0] <= ts:
        g_idx += 1
    if g_idx < len(gnum_markers):
        gnum = gnum_markers[g_idx][1]
        bvh_vs_gnum.append((gnum, dur_ms))

# Step 3: Compute average BVH update time per particle count
bvh_by_particles = defaultdict(list)
for gnum, dur_ms in bvh_vs_gnum:
    bvh_by_particles[gnum].append(dur_ms)

avg_particle_counts = []
avg_bvh_times = []

for gnum in sorted(bvh_by_particles):
    avg_particle_counts.append(gnum)
    avg_bvh_times.append(np.mean(bvh_by_particles[gnum]))

# Step 4: Compute scaled O(n log n) curve
particle_counts_array = np.array(avg_particle_counts)
nlogn = particle_counts_array * np.log2(particle_counts_array)

# Scale O(n log n) to roughly match average times for visualization
# We'll scale so max(nlogn) == max(avg_bvh_times)
scale_factor = max(avg_bvh_times) / max(nlogn)
scaled_nlogn = nlogn * scale_factor

# Step 5: Plot average BVH time and scaled O(n log n)
plt.figure(figsize=(9, 6))
plt.plot(avg_particle_counts, avg_bvh_times, marker='o', linestyle='-', label='Average BVH Update Time')
plt.plot(avg_particle_counts, scaled_nlogn, linestyle='--', color='red', label='Scaled O(n log n)')
plt.xlabel("Particle Count (gnum)")
plt.ylabel("Time (ms)")
plt.title("Average BVH Update Time vs Particle Count\nwith O(n log n) Reference")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: (Optional) print correlation
corr, _ = pearsonr(avg_particle_counts, avg_bvh_times)
print(f"Pearson correlation (averaged): {corr:.4f}")
