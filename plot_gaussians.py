import numpy as np
import matplotlib.pyplot as plt


dist1 = np.load("matching_gaussian.npy")
dist2 = np.load("non_matching_gaussian.npy")
scores1 = np.load("matching_scores.npy")
scores2 = np.load("non_matching_scores.npy")
plt.plot(np.linspace(np.min(scores1), np.max(scores1), len(dist1)), dist1)
plt.plot(np.linspace(np.min(scores2), np.max(scores2), len(dist2)), dist2)
plt.legend(["Matching", "Non-matching"])
plt.show()
