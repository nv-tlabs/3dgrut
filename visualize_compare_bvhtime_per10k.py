import os
import json
import matplotlib.pyplot as plt

# scenes = ["lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship"]
scenes = ["flowers","lego"]
colors = plt.cm.tab10.colors  # Use up to 10 distinct colors

def extract_normalized_bvh(filepath):
    bvh_events = []
    gnum_markers = []

    with open(filepath, "r") as f:
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

    # Sort both by timestamp
    bvh_events.sort()
    gnum_markers.sort()

    # Match BVH events to nearest previous gnum marker
    data = []
    g_idx = 0
    for ts, dur_ms, step in bvh_events:
        while g_idx + 1 < len(gnum_markers) and gnum_markers[g_idx + 1][0] <= ts:
            g_idx += 1
        if g_idx < len(gnum_markers):
            gnum = gnum_markers[g_idx][1]
            dur_per_10k = dur_ms / (gnum / 10_000)
            data.append((step, dur_per_10k))

    return sorted(data)

# Plotting
plt.figure(figsize=(12, 6))

for i, scene in enumerate(scenes):
    json_path = f"profile_report_{scene}.json"
    if not os.path.exists(json_path):
        print(f"Missing: {json_path}")
        continue
    bvh_data = extract_normalized_bvh(json_path)
    if bvh_data:
        steps, norm_durations = zip(*bvh_data)
        plt.plot(steps, norm_durations, label=scene, color=colors[i % len(colors)], marker='o')

plt.xlabel("Training Step")
plt.ylabel("BVH Build Time per 10K Particles (ms)")
plt.title("Normalized BVH Build Time per Step (All NeRF Synthetic Scenes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
