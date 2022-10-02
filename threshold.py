import numpy as np
from match_score import match_score
import glob
import os
import tqdm


def update_dist(list1, list2, max_score, num_bins, matching_dist, non_matching_dist):
    for file1 in tqdm.tqdm(list1):
        type1 = "_".join(os.path.basename(file1).split("_")[1:])
        for file2 in list2:
            type2 = "_".join(os.path.basename(file2).split("_")[1:])
            arr1 = np.load(file1)[0]
            arr2 = np.load(file2)[0]
            score = min(max_score, match_score(arr1, arr2))
            bin_num = int((num_bins-1) * score / max_score)
            if type1 == type2:
                matching_dist[bin_num] += 1
            else:
                non_matching_dist[bin_num] += 1


def get_threshold(data_dir, max_score=25, num_bins=2048):
    types = ["Intonation", "Phoneme", "Sentence", "Stress"]
    names = set(["jeeva", "anju"])
    matching_dist = np.zeros(num_bins)
    non_matching_dist = np.zeros(num_bins)
    for t in types[:1]:
        print(t)
        pattern = f"{data_dir}/*_{t}*.npy"
        results = sorted(glob.glob(pattern))
        name_wise = {}
        for result in results:
            filename = os.path.basename(result)
            name = filename.split("_")[0]
            if name not in names: continue
            if name not in name_wise:
                name_wise[name] = []
            name_wise[name].append(result)
        names = sorted(list(name_wise.keys()))
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                update_dist(name_wise[names[i]], name_wise[names[j]], max_score, num_bins,
                            matching_dist, non_matching_dist)
    matching_dist = matching_dist/np.sum(matching_dist)
    non_matching_dist = non_matching_dist/np.sum(non_matching_dist)
    np.save("matching_dist.npy", matching_dist)
    np.save("non_matching_dist.npy", non_matching_dist)
    intersection = -1
    for i in range(num_bins):
        if np.allclose(matching_dist[i], non_matching_dist[i]):
            intersection = i
    th_inter = max_score * intersection / (num_bins-1)
    scores = np.linspace(0, 1, num_bins) * max_score
    th_mean = (np.sum(scores * matching_dist) + np.sum(scores * non_matching_dist)) / 2
    return th_inter, th_mean


if __name__ == "__main__":
    data_dir = "data/sentencewise_features"
    get_threshold(data_dir)
