import numpy as np
from match_score import match_score
import glob
import os
#import tqdm
from scipy.stats import norm
import sys
import multiprocessing as mp
import pandas as pd

'''
    Function which is to be executed by many processes parallely
    args: arguments to the function
    args[0]: filename of 1st file
    args[1]: filename of 2nd file
    args[2]: version of using match_score
'''
def get_score(args):
    arr1 = np.load(args[0])[0]
    arr2 = np.load(args[1])[0]
    version = args[2]
    return match_score(arr1, arr2, version)


def update_dist(list1, list2, matching_scores, non_matching_scores, num_cores, version):
    #N_arguments passed are name_wise[names[i]], name_wise[names[j]],matching_scores, non_matching_scores, num_cores, version

    args = [] # args for each process are stored in this list (file1, file2, version)
    same = [] # tells whether the score should be added in matching scores or non_matching scores
    for file1 in list1:
        type1 = "_".join(os.path.basename(file1).split("_")[1:]) # word related information #N_ what information ?
        for file2 in list2:
            type2 = "_".join(os.path.basename(file2).split("_")[1:]) # word related information
            if type1 == type2: # The utteranances in both the files is of the same word
                same.append(True)
            else:
                same.append(False)
            args.append((file1, file2, version))
    with mp.Pool(num_cores) as P:
        i = 0
        for score in P.map_async(get_score, args).get(): # parallelizing calculation of match scores between many pairs of files
            if same[i]:
                matching_scores.append(score)
            else:
                non_matching_scores.append(score)
            i += 1


def solve(m1,m2,std1,std2):
    # the method id to solve the gaussian eqautions to find the roots where ever it 
    # intersects , i.e give roots of the equation.
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    roots = np.roots([a,b,c])
    for i in range(len(roots)):
        if roots[i] >= np.min([m1, m2]) and roots[i] <= np.max([m1, m2]):
            return roots[i]


def get_threshold(data_dir, num_cores=1, version=1):
    print(f"using {num_cores} cores")
    types = ["Intonation", "Phoneme", "Sentence", "Stress"]
    
    df = pd.read_csv("data/data_copy.csv", header=None) # data which tells whether a file contains intelligible utterance or not.
    df = df.values.tolist()
    df = {x[0].lower(): x[1] for x in df} # converting dataframe to a dict with key: val as filename: not_intelligible.
    # names = set(["jeeva", "anju", "bharati"])
    
    matching_scores = []
    non_matching_scores = []
    
    for t in types[:1]: # to run on entire data: types[:1] to types
        print(t)
        pattern = f"{data_dir}/*_{t}_L4*.npy" # to run on entire data: f"{data_dir}/*_{t}_L4*.npy" to f"{data_dir}/*_{t}*.npy"
        results = sorted(glob.glob(pattern)) #N_what are results ? 
        name_wise = {} # stores the files related to each speaker.
        for result in results:
            filename = os.path.basename(result)
            name = filename.split("_")[0]
            # if name not in names: continue # to run on entire data: comment this line
            if name not in name_wise:
                name_wise[name] = []
            if df[filename.lower().split(".")[0]] == 0: # include onnly if it is intelligible
                name_wise[name].append(result)
        
        names = sorted(list(name_wise.keys()))
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                print(names[i], names[j])
                #N_the kl d
                update_dist(name_wise[names[i]], name_wise[names[j]],
                            matching_scores, non_matching_scores, num_cores, version)
    
    matching_scores = np.array(matching_scores)
    non_matching_scores = np.array(non_matching_scores)
    np.save(f"results/matching_scores{version}.npy", matching_scores)
    np.save(f"results/non_matching_scores{version}.npy", non_matching_scores)
    
    m_mean, m_std = norm.fit(matching_scores)
    nm_mean, nm_std = norm.fit(non_matching_scores)
    
    matching_x = np.linspace(np.min(matching_scores), np.max(matching_scores), 1000)
    non_matching_x = np.linspace(np.min(non_matching_scores), np.max(non_matching_scores), 1000)
    matching_dist = norm.pdf(matching_x, m_mean, m_std)
    non_matching_dist = norm.pdf(non_matching_x, nm_mean, nm_std)
    np.save(f"results/matching_gaussian{version}.npy", matching_dist)
    np.save(f"results/non_matching_gaussian{version}.npy", non_matching_dist)
    
    #here we are calculating the average of both standard devantion means to find threshold 1
    th_mean = (nm_mean+m_mean)/2
    #here we are finding the second threshold
    th_inter = solve(m_mean,nm_mean,m_std,nm_std)
    # saving results in a txt file
    with open(f"results/threshold_{version}.txt", "w+") as file:
        file.write(f"{th_inter} {th_mean}\n")
    return th_inter, th_mean 


if __name__ == "__main__":
    #data_dir = "data/sentencewise_features" # the path to the directory which contains our vector files
    data_dir="/media/nayan/z/wav2vec2_implementaions/wav2vec2_features_for_student_dataset/sentencewise_features"
    num_cores = os.cpu_count() # number of cores to use (number of processes to run in parallel (max possible: number of cores on the cpu))
    if len(sys.argv) > 1:
        num_cores = min(num_cores, int(sys.argv[1]))
    version = 1
    if len(sys.argv) > 2:
        version = int(sys.argv[2])
    get_threshold(data_dir, num_cores, version)
