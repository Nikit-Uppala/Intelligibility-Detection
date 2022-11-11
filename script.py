import numpy as np
import sys
from scipy.stats import norm
from threshold_gaussian import solve


def normalized_scores(m_scores, nm_scores):
    mini = min(np.min(m_scores), np.min(nm_scores))
    maxi = max(np.max(m_scores), np.max(nm_scores))
    m_norm_scores = (m_scores-mini) / (maxi-mini)
    nm_norm_scores = (nm_scores-mini) / (maxi-mini)
    return m_norm_scores, nm_norm_scores

def clip(scores):
    std = np.std(scores)
    mean = np.mean(scores)
    clipped_scores = scores[(scores > mean - 3 * std) & (scores < mean + 3 * std)]
    return clipped_scores


def get_thresholds(m_scores, nm_scores):
    m_mean, m_std = norm.fit(m_scores)
    nm_mean, nm_std = norm.fit(nm_scores)
    th_mean = (m_mean + nm_mean) / 2
    th_inter = solve(m_mean, nm_mean, m_std, nm_std)
    return th_mean, th_inter


def main():
    m_scores = np.load(f"results/matching_scores{version}.npy")
    nm_scores = np.load(f"results/non_matching_scores{version}.npy")
    
    # normalized scores
    m_norm_scores, nm_norm_scores = normalized_scores(m_scores, nm_scores)
    np.save(f"results/matching_normalized_scores{version}.npy", m_norm_scores)
    np.save(f"results/non_matching_normalized_scores{version}.npy", nm_norm_scores)
    th_mean, th_inter = get_thresholds(m_norm_scores, nm_norm_scores)
    with open(f"results/threshold_normalized{version}.txt", "w") as file:
        file.write(f"{th_inter} {th_mean}")

    # clipping scores between -3 * sigma and 3 * sigma
    m_clip_scores = clip(m_scores)
    nm_clip_scores = clip(nm_scores)
    np.save(f"results/matching_clipped_scores{version}.npy", m_clip_scores)
    np.save(f"results/non_matching_clipped_scores{version}.npy", nm_clip_scores)
    th_mean, th_inter = get_thresholds(m_clip_scores, nm_clip_scores)
    with open(f"results/threshold_clipped{version}.txt", "w") as file:
        file.write(f"{th_inter} {th_mean}")


if __name__ == '__main__':
    version = 1
    if len(sys.argv) > 1:
        version = int(sys.argv[1])
    main()