import torch
from pathlib import Path

class_dists = Path("class_dists")

cycle0 = torch.load(class_dists / "dist_cycle_0.pt")
cycle1 = torch.load(class_dists / "dist_cycle_1.pt")
cycle2 = torch.load(class_dists / "dist_cycle_2.pt")
cycle3 = torch.load(class_dists / "dist_cycle_3.pt")
cycle4 = torch.load(class_dists / "dist_cycle_4.pt")
cycle5 = torch.load(class_dists / "dist_cycle_0.pt")
cycle6 = torch.load(class_dists / "dist_cycle_1.pt")
cycle7 = torch.load(class_dists / "dist_cycle_2.pt")
cycle8 = torch.load(class_dists / "dist_cycle_3.pt")
cycle9 = torch.load(class_dists / "dist_cycle_4.pt")
cycle10 = torch.load(class_dists / "dist_cycle_0.pt")
cycle11 = torch.load(class_dists / "dist_cycle_1.pt")
cycle12 = torch.load(class_dists / "dist_cycle_2.pt")
cycle13 = torch.load(class_dists / "dist_cycle_3.pt")
cycle14 = torch.load(class_dists / "dist_cycle_4.pt")
cycle15 = torch.load(class_dists / "dist_cycle_0.pt")
cycle16 = torch.load(class_dists / "dist_cycle_1.pt")
cycle17 = torch.load(class_dists / "dist_cycle_2.pt")
cycle18 = torch.load(class_dists / "dist_cycle_3.pt")
cycle19 = torch.load(class_dists / "dist_cycle_4.pt")
cycle20 = torch.load(class_dists / "dist_cycle_0.pt")
cycle21 = torch.load(class_dists / "dist_cycle_1.pt")
cycle22 = torch.load(class_dists / "dist_cycle_2.pt")
cycle23 = torch.load(class_dists / "dist_cycle_3.pt")
cycle24 = torch.load(class_dists / "dist_cycle_4.pt")
cycle25 = torch.load(class_dists / "dist_cycle_4.pt")

labels = torch.load(class_dists / "class_names_cycle_0.pt")
labels = [label.split(",")[0] for label in labels.values()]

import matplotlib.pyplot as plt

x = torch.arange(0, 100)

plt.figure(figsize=(15, 5))

plt.bar(labels, cycle4, label="cycle 4")
plt.bar(labels, cycle3, label="cycle 3")
plt.bar(labels, cycle2, label="cycle 2")
plt.bar(labels, cycle1, label="cycle 1")
plt.bar(labels, cycle0, label="cycle 0")
plt.xticks(rotation=90, ha="center", fontsize=8)
plt.tight_layout(pad=3)


plt.legend()
plt.savefig("./class_dists.pdf", format="pdf")
plt.close()
