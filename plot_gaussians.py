import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from scipy.stats import norm

def main():
    """This function is used to plot the gaussians, with respect to the arguments given, the data to plot gaussian is saved in
    results folder.  
    """
    matching_files = sorted(glob.glob(f"results/matching_{'' if scores_type is None else scores_type+'_'}scores*.npy"))
    non_matching_files = sorted(glob.glob(f"results/non_matching_{'' if scores_type is None else scores_type+'_'}scores*.npy"))
    assert len(matching_files) == len(non_matching_files)
    fig, ax = plt.subplots(len(matching_files), figsize=(14, 5 * len(matching_files)))
    print(len(matching_files))
    for i in range(len(matching_files)):
        matching_scores = np.load(matching_files[i])
        non_matching_scores = np.load(non_matching_files[i])
        m_mean, m_std = norm.fit(matching_scores)
        nm_mean, nm_std = norm.fit(non_matching_scores)
        m_x = np.linspace(matching_scores.min(), matching_scores.max(), 1000)
        nm_x = np.linspace(non_matching_scores.min(), non_matching_scores.max(), 1000)
        m_pdf = norm.pdf(m_x, m_mean, m_std)
        nm_pdf = norm.pdf(nm_x, nm_mean, nm_std)
        plt.plot(m_x, m_pdf, nm_x, nm_pdf)
        plt.legend(["matching", "non_matching"])
    plt.savefig("results/gaussians_fit.png")
    plt.show()


if __name__ == "__main__":
    """The input to this include two type, normalised and clipped, this is done as the paper doe calculation on probability distribution
    whereas the one we had was, features and they could be negative.
    """
    scores_type = None
    if len(sys.argv) > 1:
        # Three types of arguments that is none,normalized, clipped
        # None will use the same data as provided, 
        # normalised will normalise the data between 0-1
        # clipped will clip the data to mean-3*sd and mean+3*sd
        scores_type = sys.argv[1] 
    main()
