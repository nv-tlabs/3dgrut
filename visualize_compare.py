import json
import os
import matplotlib.pyplot as plt

scenes = ["lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship"]
colors = plt.cm.tab10.colors  # Use up to 10 distinct colors

def extract_bvh_times(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                nvtx = obj.get("NvtxEvent", {})
                name = nvtx.get("Text", "")
                if name.startswith("train_") and name.endswith("_bvh"):
                    step = int(name.split("_")[1])
                    start_ns = int(nvtx["Timestamp"])
                    end_ns = int(nvtx["EndTimestamp"])
                    duration_ms = (end_ns - start_ns) / 1e6
                    data.append((step, duration_ms))
            except:
                continue
    return sorted(data)

plt.figure(figsize=(12, 6))

for i, scene in enumerate(scenes):
    json_path = f"profile_report_{scene}.json"
    if not os.path.exists(json_path):
        print(f"Missing: {json_path}")
        continue
    bvh_data = extract_bvh_times(json_path)
    if bvh_data:
        steps, durations = zip(*bvh_data)
        plt.plot(steps, durations, label=scene, color=colors[i % len(colors)], marker='o')

plt.xlabel("Training Step")
plt.ylabel("BVH Build Time (ms)")
plt.title("BVH Build Time per Training Step Across NeRF Synthetic Scenes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
