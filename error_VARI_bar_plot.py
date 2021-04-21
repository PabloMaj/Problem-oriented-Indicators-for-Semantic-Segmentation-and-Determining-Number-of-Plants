import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(figsize=(10, 5))

VI_names = ["ExG", "Optimised\nlinear", "VDVI", "Optimised\nfraction", "Mask R-CNN"]
y_values = dict()
y_values["flowering"] = [1.21, 2.93, 7.55, 6.98, 3.95]
y_values["mature"] = [11.88, 5.05, 0.32, 0.40, 3.04]
y_values["before harvest"] = [5.47, 0.06, 8.68, 6.78, 2.51]
y_pos = np.arange(len(VI_names))
height = 0.2

ax.barh(y_pos - height, y_values["flowering"], align='center', height=height, zorder=3, edgecolor="k")
ax.barh(y_pos, y_values["mature"], align='center', height=height, zorder=3, edgecolor="k")
ax.barh(y_pos + height, y_values["before harvest"], align='center', height=height, zorder=3, edgecolor="k")
ax.set_yticks(y_pos)
ax.set_yticklabels(VI_names)
ax.set_xlabel("MAPE of mean VARI for plants [%]")
ax.grid()
ax.grid(zorder=0)
fig.tight_layout()

plt.legend(["flowering", "mature", "before harvest"])

plt.savefig("MAPE_of_Mean_VARI.png", dpi=300)
# plt.show()
