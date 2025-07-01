import json
import matplotlib.pyplot as plt

gnum_data = []

with open("profile_report.json", "r") as f:
    for line in f:
        try:
            obj = json.loads(line)
            nvtx = obj.get("NvtxEvent", {})
            name = nvtx.get("Text", "")
            if name.startswith("gnum_"):
                gnum = int(name.split("_")[-1])
                timestamp = int(nvtx["Timestamp"])  # You can also use step alignment if available
                gnum_data.append((timestamp, gnum))
        except:
            continue

# Sort and plot
gnum_data.sort()
if gnum_data:
    timestamps, gnums = zip(*gnum_data)
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, gnums, marker='x', linestyle='-', color='purple')
    plt.xlabel("Timestamp (ns)")
    plt.ylabel("gNum (Number of Gaussians)")
    plt.title("gNum Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No gnum NVTX markers found.")
