import json
import matplotlib.pyplot as plt
import numpy as np

with open("summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

overall = summary["overall"]
by_cat = summary["by_category"]
conf = summary["engine_confusion"]

# 1) Overall metrics
overall_labels = [
    "Token F1", "Routing Acc", "Hit@1", "Hit@3", "Hit@5"
]
overall_values = [
    overall["token_f1"],
    overall["routing_accuracy"],
    overall["source_hit_at_1"],
    overall["source_hit_at_3"],
    overall["source_hit_at_5"],
]

plt.figure(figsize=(8, 5))
plt.bar(overall_labels, overall_values)
plt.ylabel("Percentage")
plt.title("Overall Evaluation Metrics")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("chart_overall_metrics.png", dpi=200)
plt.close()

# 2) By-category grouped bars
categories = list(by_cat.keys())
f1_vals = [by_cat[c]["token_f1"] for c in categories]
route_vals = [by_cat[c]["routing_accuracy"] for c in categories]
hit5_vals = [by_cat[c]["source_hit_at_5"] for c in categories]

x = np.arange(len(categories))
w = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - w, f1_vals, width=w, label="Token F1")
plt.bar(x, route_vals, width=w, label="Routing Accuracy")
plt.bar(x + w, hit5_vals, width=w, label="SourceHit@5")
plt.xticks(x, categories, rotation=30)
plt.ylabel("Percentage")
plt.title("Metrics by Category")
plt.legend()
plt.tight_layout()
plt.savefig("chart_by_category.png", dpi=200)
plt.close()

# 3) Average latency by category
lat_vals = [by_cat[c]["avg_latency_ms"] for c in categories]

plt.figure(figsize=(10, 5))
plt.bar(categories, lat_vals)
plt.xticks(rotation=30)
plt.ylabel("Milliseconds")
plt.title("Average Latency by Category")
plt.tight_layout()
plt.savefig("chart_latency_by_category.png", dpi=200)
plt.close()

# 4) Router confusion matrix
pairs = {}
row_labels = set()
col_labels = set()

for k, v in conf.items():
    left, right = [s.strip() for s in k.split("->")]
    pairs[(left, right)] = v
    row_labels.add(left)
    col_labels.add(right)

row_labels = sorted(row_labels)
col_labels = sorted(col_labels)

mat = np.zeros((len(row_labels), len(col_labels)), dtype=int)
for i, r in enumerate(row_labels):
    for j, c in enumerate(col_labels):
        mat[i, j] = pairs.get((r, c), 0)

plt.figure(figsize=(8, 6))
plt.imshow(mat, aspect="auto")
plt.colorbar(label="Count")
plt.xticks(np.arange(len(col_labels)), col_labels, rotation=30)
plt.yticks(np.arange(len(row_labels)), row_labels)
plt.title("Engine Confusion Matrix")

for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        plt.text(j, i, str(mat[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig("chart_engine_confusion.png", dpi=200)
plt.close()

print("Done: 4 charts saved.")