import os
import json
import matplotlib.pyplot as plt

scenes = ["lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship"]
colors = plt.cm.tab10.colors

def extract_avg_bvh_per10k(filepath, step_start=600, step_end=1000):
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
                        step = int(name.split("_")[1])
                        if step < step_start or step >= step_end:
                            continue
                        end_ts = int(evt["EndTimestamp"])
                        dur_ms = (end_ts - ts) / 1e6
                        bvh_events.append((ts, dur_ms, step))
                    elif name.startswith("gnum_"):
                        gnum = int(name.split("_")[1])
                        gnum_markers.append((ts, gnum))
            except:
                continue

    bvh_events.sort()
    gnum_markers.sort()

    g_idx = 0
    per10k_durations = []

    for ts, dur_ms, step in bvh_events:
        while g_idx + 1 < len(gnum_markers) and gnum_markers[g_idx + 1][0] <= ts:
            g_idx += 1
        if g_idx < len(gnum_markers):
            gnum = gnum_markers[g_idx][1]
            if gnum > 0:
                dur_per_10k = dur_ms / (gnum / 10_000)
                per10k_durations.append(dur_per_10k)

    if per10k_durations:
        return sum(per10k_durations) / len(per10k_durations)
    else:
        return None

# Collect averages
scene_averages = []

for scene in scenes:
    path = f"profile_report_{scene}.json"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    avg = extract_avg_bvh_per10k(path)
    if avg is not None:
        scene_averages.append((scene, avg))
    else:
        print(f"No valid data for scene: {scene}")

# Bar chart
plt.figure(figsize=(10, 6))
scenes_to_plot, averages_to_plot = zip(*scene_averages)
bars = plt.bar(scenes_to_plot, averages_to_plot, color=colors[:len(scenes_to_plot)])

plt.ylabel("Avg BVH Build Time per 10K Particles (ms)")
plt.title("Average BVH Build Time (Steps 600–999)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
