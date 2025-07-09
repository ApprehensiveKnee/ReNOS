import matplotlib.pyplot as plt
import numpy as np

stats = np.load("statistics.npy", allow_pickle=True).item()
mean_values = np.array(stats["mdn"])
std_values = np.array(stats["std"])
best_values = np.array([stats["best"][i][2] for i in range(len(stats["best"]))])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# add mean line
ax.plot(mean_values, label="iteration mean", color = "lightseagreen")
# fill between the -std and +std
ax.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2, color = "mediumturquoise")
# add best line
ax.plot(best_values, label="iteration best", color = "lightcoral")
ax.set_xlim(0, len(mean_values))
ax.tick_params(direction='in')
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.xaxis.set_tick_params(width=2)
ax.yaxis.set_tick_params(width=2)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.legend(fontsize=14)
plt.xlabel("Iterations", fontdict={"size": 14})
plt.ylabel("Latency estimation", fontdict={"size": 14})
plt.title("ACO convergence", fontdict={"size": 16})
plt.show()

fig.savefig("visual/convergence.png", dpi=300)
