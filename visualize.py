import json
import matplotlib.pyplot as plt

bvh_data = []

with open("profile_report.json", "r") as f:
    for line in f:
        try:
            obj = json.loads(line)
            nvtx = obj.get("NvtxEvent", {})
            name = nvtx.get("Text", "")
            if name.startswith("train_") and name.endswith("_bvh"):
                step = int(name.split("_")[1])
                start_ns = int(nvtx["Timestamp"])
                end_ns = int(nvtx["EndTimestamp"])
                duration_ms = (end_ns - start_ns) / 1e6  # convert ns to ms
                bvh_data.append((step, duration_ms))
        except Exception as e:
            continue

# Sort and plot
bvh_data.sort()
if bvh_data:
    steps, durations = zip(*bvh_data)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, durations, marker='o', linestyle='-', color='blue')
    plt.xlabel("Training Step")
    plt.ylabel("BVH Build Time (ms)")
    plt.title("BVH Build Time per Training Iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No matching BVH events found.")
