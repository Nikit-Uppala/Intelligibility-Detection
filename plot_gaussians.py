import numpy as np
import matplotlib.pyplot as plt
import sys


version = 1
if len(sys.argv) > 1:
    version = int(sys.argv[1])
dist1 = np.load(f"matching_gaussian{version}.npy")
dist2 = np.load(f"non_matching_gaussian{version}.npy")
scores1 = np.load(f"matching_scores{version}.npy")
scores2 = np.load(f"non_matching_scores{version}.npy")
print(np.min(scores1), np.max(scores2))
plt.plot(np.linspace(np.min(scores1), np.max(scores1), len(dist1)), dist1)
plt.plot(np.linspace(np.min(scores2), np.max(scores2), len(dist2)), dist2)
plt.legend(["Matching", "Non-matching"])
plt.show()
