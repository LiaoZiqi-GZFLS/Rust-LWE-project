import matplotlib.pyplot as plt

labels = ['gate_apply', 'rotate_gate', 'others']
before = [78, 12, 10]
after = [18, 58, 24]

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axs[0].barh(labels, before, color=['#e41a1c', '#ff7f00', '#984ea3'])
axs[0].set_title("Before Optimization")
axs[0].set_xlim(0, 100)
axs[0].set_xlabel("CPU Time (%)")

axs[1].barh(labels, after, color=['#377eb8', '#4daf4a', '#a65628'])
axs[1].set_title("After Optimization")
axs[1].set_xlim(0, 100)
axs[1].set_xlabel("CPU Time (%)")

fig.suptitle("Function Time Distribution in FHE Simulation", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("fhe_flame_chart_en.png", dpi=300)
plt.show()
