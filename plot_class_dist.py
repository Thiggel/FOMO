import torch
from pathlib import Path

class_dists = Path("class_dists")

cycle0 = torch.load(class_dists / "dist_cycle_0.pt")
cycle1 = torch.load(class_dists / "dist_cycle_1.pt")
cycle2 = torch.load(class_dists / "dist_cycle_2.pt")
cycle3 = torch.load(class_dists / "dist_cycle_3.pt")
cycle4 = torch.load(class_dists / "dist_cycle_4.pt")

import matplotlib.pyplot as plt

x = torch.arange(0, 100)

plt.bar(x, cycle4, label="cycle 4")
plt.bar(x, cycle3, label="cycle 3")
plt.bar(x, cycle2, label="cycle 2")
plt.bar(x, cycle1, label="cycle 1")
plt.bar(x, cycle0, label="cycle 0")

plt.legend()
plt.show()
