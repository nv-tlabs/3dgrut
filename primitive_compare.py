import os
import json
import matplotlib.pyplot as plt

# Match these to your exported primitives and filenames
primitives = ["icosahedron", "octahedron", "trihexa", "trisurfel", "tetrahedron", "sphere", "diamond"]
primitive_labels = {
    "icosahedron": "Icosahedron",
    "octahedron": "Octahedron",
    "trihexa": "TriHexa",
    "trisurfel": "TriSurfel",
    "tetrahedron": "Tetrahedron",
    "sphere": "Sphere",
    "diamond": "Diamond"
}

def extract_primitive_build_times(json_path, label):
    durations = []
    with open(json_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "NvtxEvent" in obj:
                    evt = obj["NvtxEvent"]
                    text = evt.get("Text", "")
                    if text == f"Primitive: {label}":
                        start = int(evt["Timestamp"])
                        end = int(evt["EndTimestamp"])
                        duration_ms = (end - start) / 1e6
                        durations.append(duration_ms)
            except Exception:
                continue
    return durations

# Collect average times
average_times = []

for primitive in primitives:
    label = primitive_labels[primitive]
    json_path = f"traces/primitive_comparison/{primitive}/profile_report_{primitive}.json"
    if not os.path.exists(json_path):
        print(f"Missing: {json_path}")
        continue

    times = extract_primitive_build_times(json_path, label)
    if times:
        avg_time = sum(times) / len(times)
        average_times.append((label, avg_time))
    else:
        print(f"No build times found for {label}")

# Sort by average time (optional)
average_times.sort(key=lambda x: x[1], reverse=True)

# Plot
labels, values = zip(*average_times)
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color='steelblue')
plt.ylabel("Avg Primitive Build Time (ms)")
plt.title("Average BVH Build Time by Primitive Type")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate bars with exact values
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
