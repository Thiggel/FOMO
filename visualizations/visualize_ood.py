import numpy as np
import matplotlib.pyplot as plt

files = [
    "ood_logs/0/scores_ood.npy",
    "ood_logs/1/scores_ood.npy",
    "ood_logs/2/scores_ood.npy",
    "ood_logs/3/scores_ood.npy",
]

labels = [
    "Cycle 0",
    "Cycle 1",
    "Cycle 2",
    "Cycle 3",
]

for index, file in enumerate(files):
    data = np.load(file)

    # plot the distances from lowest to highest
    data = np.sort(data)

    x = np.arange(len(data)) / len(data)

    plt.plot(x, data, label=labels[index])

plt.xlabel("Percentage of data")
plt.ylabel("Distance to Nearest Neighbors")
plt.legend()
plt.savefig("ood_logs/ood.pdf", format="pdf", bbox_inches="tight")
