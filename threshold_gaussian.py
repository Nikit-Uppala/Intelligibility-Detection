import numpy as np
from match_score import match_score
import glob
import os
import tqdm
from scipy.stats import norm
import sys


def update_dist(list1, list2, matching_scores, non_matching_scores, version):
    for file1 in list1:
        type1 = "_".join(os.path.basename(file1).split("_")[1:])
        for file2 in tqdm.tqdm(list2):
            type2 = "_".join(os.path.basename(file2).split("_")[1:])
            arr1 = np.load(file1)[0]
            arr2 = np.load(file2)[0]
            score = match_score(arr1, arr2, version)
            if type1 == type2:
                matching_scores.append(score)
            else:
                non_matching_scores.append(score)


def solve(m1,m2,std1,std2):
    # the method id to solve the gaussian eqautions to find the roots where ever it 
    # intersects , i.e give roots of the equation.
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])


def get_threshold(data_dir, version=1):
    types = ["Intonation", "Phoneme", "Sentence", "Stress"]
    names = set(["jeeva", "anju"])
    matching_scores = []
    non_matching_scores = []
    for t in types[:1]: # to run on entire data: types[:1] to types
        print(t)
        pattern = f"{data_dir}/*_{t}_L4*.npy" # to run on entire data: f"{data_dir}/*_{t}_L4*.npy" to f"{data_dir}/*_{t}*.npy"
        results = sorted(glob.glob(pattern))
        name_wise = {}
        for result in results:
            filename = os.path.basename(result)
            name = filename.split("_")[0]
            if name not in names: continue # to run on entire data: comment this line
            if name not in name_wise:
                name_wise[name] = []
            name_wise[name].append(result)
        names = sorted(list(name_wise.keys()))
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                print(len(name_wise[names[i]]), len(name_wise[names[j]]))
                update_dist(name_wise[names[i]], name_wise[names[j]],
                            matching_scores, non_matching_scores, version)
    matching_scores = np.array(matching_scores)
    non_matching_scores = np.array(non_matching_scores)
    np.save(f"matching_scores{version}.npy", matching_scores)
    np.save(f"non_matching_scores{version}.npy", non_matching_scores)
    th_mean = (np.mean(matching_scores) + np.mean(non_matching_scores))/2
    m_mean, m_std = norm.fit(matching_scores)
    nm_mean, nm_std = norm.fit(non_matching_scores)
    matching_x = np.linspace(np.min(matching_scores), np.max(matching_scores), 1000)
    non_matching_x = np.linspace(np.min(non_matching_scores), np.max(non_matching_scores), 1000)
    matching_dist = norm.pdf(matching_x, m_mean, m_std)
    non_matching_dist = norm.pdf(non_matching_x, nm_mean, nm_std)
    np.save(f"matching_gaussian{version}.npy", matching_dist)
    np.save(f"non_matching_gaussian{version}.npy", non_matching_dist)
    #here we are calculating the average of both standard devantion means to find threshold 1
    th_mean = (nm_mean+m_mean)/2
    #here we are finding the second threshold
    th_inter = solve(m_mean,nm_mean,m_std,nm_std)[-1]
    return th_inter, th_mean 


if __name__ == "__main__":
    data_dir = "data/sentencewise_features"
    version = 1
    if len(sys.argv) > 1:
        version = int(sys.argv[1])
    get_threshold(data_dir, version)
