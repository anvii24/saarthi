
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation/results/graph.csv")

metrics = ["P@5", "MAP", "MRR", "nDCG@5"]

for metric in metrics:
    plt.figure(figsize=(9, 5))
    plt.bar(df["method"], df[metric])
    plt.title(f"Graph Enhancement Comparison: {metric}")
    plt.ylabel(metric)
    plt.xlabel("Retrieval Method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"evaluation/results/graph_{metric.replace('@', '_')}.png")
    plt.close()

print("Graph comparison charts saved.")
