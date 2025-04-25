import matplotlib.pyplot as plt
import os

#%% ========== Visualization ==========
def plot_metrics(metrics_dict):
    labels = list(metrics_dict["Gradient Ascent Unlearning"].keys())
    title = "Unlearning Methods Comparison"
    x = range(len(labels))
    
    plt.figure(figsize=(12, 6))
    for method, scores in metrics_dict.items():
        values = [float(scores[label].split()[0]) if isinstance(scores[label], str) else float(scores[label]) for label in labels]
        plt.plot(x, values, marker='o', label=method)

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/100/{title.replace(' ', '_').lower()}_similarity.png")
    plt.close()
    
#%%