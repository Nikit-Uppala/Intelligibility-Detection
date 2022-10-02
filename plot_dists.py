import numpy as np
import matplotlib.pyplot as plt


dist1 = np.load("matching_dist.npy")
dist2 = np.load("non_matching_dist.npy")
plt.plot(dist1)
plt.plot(dist2)
plt.legend(["Matching", "Non-matching"])
plt.show()
