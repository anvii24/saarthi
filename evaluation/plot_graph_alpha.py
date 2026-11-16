
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation/results/graph_alpha.csv")

metrics = ["P@5", "MAP", "MRR", "nDCG@5"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.plot(df["alpha"], df[metric], marker="o")
    plt.title(f"Topic PageRank Alpha Sensitivity: {metric}")
    plt.xlabel("Alpha (Dense Retrieval Weight)")
    plt.ylabel(metric)
    plt.xticks(df["alpha"])
    plt.tight_layout()
    plt.savefig(
        f"evaluation/results/graph_alpha_{metric.replace('@', '_')}.png"
    )
    plt.close()

print("Alpha sensitivity charts saved.")
