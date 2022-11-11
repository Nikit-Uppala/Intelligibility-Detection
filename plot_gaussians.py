import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from scipy.stats import norm


def main():
    matching_files = sorted(glob.glob(f"results/matching_{'' if scores_type is None else scores_type+'_'}scores*.npy"))
    non_matching_files = sorted(glob.glob(f"results/non_matching_{'' if scores_type is None else scores_type+'_'}scores*.npy"))
    assert len(matching_files) == len(non_matching_files)
    versions = (
        "Taking absolute values",
        "Taking only positive values",
        "Replacing negative with small values (near zero)",
        "Normalizing between 0 and 1"
    )

    fig, ax = plt.subplots(len(matching_files), figsize=(14, 5 * len(matching_files)))
    for i in range(len(matching_files)):
        matching_scores = np.load(matching_files[i])
        non_matching_scores = np.load(non_matching_files[i])
        m_mean, m_std = norm.fit(matching_scores)
        nm_mean, nm_std = norm.fit(non_matching_scores)
        m_x = np.linspace(matching_scores.min(), matching_scores.max(), 1000)
        nm_x = np.linspace(non_matching_scores.min(), non_matching_scores.max(), 1000)
        m_pdf = norm.pdf(m_x, m_mean, m_std)
        nm_pdf = norm.pdf(nm_x, nm_mean, nm_std)
        if len(matching_files) == 1:
            ax.plot(m_x, m_pdf, nm_x, nm_pdf)
            ax.set_title(versions[i])
            ax.legend(["matching", "non_matching"])
        else:
            ax[i].plot(m_x, m_pdf, nm_x, nm_pdf)
            ax[i].set_title(versions[i])
            ax[i].legend(["matching", "non_matching"])
    plt.savefig("results/gaussians_fit.png")
    plt.show()


if __name__ == "__main__":
    scores_type = None
    if len(sys.argv) > 1:
        scores_type = sys.argv[1] # normalized, clipped
    main()
