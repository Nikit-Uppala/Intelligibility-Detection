from match_score import match_score
from scipy.stats import norm
import numpy as np
import glob
import tqdm
import sys
import os


def update_dist(list1, list2, matching_scores, non_matching_scores, version):
    """update the matching and non matching distribution.

    Args:
        list1 (list): list of 1st features.
        list2 (list): list of 2nd features.
        matching_scores (list): list for storing matching score.
        non_matching_scores (list): list for storing the non matching score.
        version (int): see match source for more information. Defaults to 1.
    """
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
    """Used methamatical formula to calculate the intersection.

    Args:
        m1 (float): mean of 1st distribution.
        m2 (float): mean of 2nd distribution.
        std1 (float): standard distribution of 1st distribution.
        std2 (float): standard distribution of 2nd distribution.

    Returns:
       float : intersection point
    """
    # the method id to solve the gaussian eqautions to find the roots where ever it 
    # intersects , i.e give roots of the equation.
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    roots = np.roots([a,b,c])
    for i in range(len(roots)):
        if roots[i] >= np.min([m1, m2]) and roots[i] <= np.max([m1, m2]):
            return roots[i]


def get_threshold(data_dir, version=1):
    """This is used to calculate the Threshold

    Args:
        data_dir (string): path of the directory containing data source directory.
        version (int, optional): see match source for more information. Defaults to 1.

    Returns:
        tuple: (threshold interaction, threshold mean)
    """
    # There were four types of files as given below and the nomenclature of them was also like that.
    '''Change the type and name of the files that you want to include in the threshold calculation
    '''
    types = ["Intonation", "Phoneme", "Sentence", "Stress"]
    names = list(["jeeva", "anju"])
    matching_scores = []
    non_matching_scores = []
    for t in types:
        print(t)
        # the below line finds the files of type t and this is used to extract the file path and load them
        pattern = f"{data_dir}/*_{t}_L4*.npy" 
        results_other = sorted(glob.glob(pattern))
        name_wise = {}
        for result in results_other:
            filename = os.path.basename(result)
            name = filename.split("_")[0]
            # to run on entire data: comment this line
            # if name not in names: continue
            if name not in name_wise:
                name_wise[name] = []
            name_wise[name].append(result)
        '''Remove the below names, if you don't want all the names to be considered comment the below line
        '''
        # names = sorted(list(name_wise.keys()))
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                try: 
                    print(len(name_wise[names[i]]), len(name_wise[names[j]]))
                    update_dist(name_wise[names[i]], name_wise[names[j]],
                            matching_scores, non_matching_scores, version)
                except: pass
    
    matching_scores = np.array(matching_scores)
    non_matching_scores = np.array(non_matching_scores)
    # the calculated scores are saved below
    np.save(f"results_other/matching_scores{version}.npy", matching_scores)
    np.save(f"results_other/non_matching_scores{version}.npy", non_matching_scores)
    
    m_mean, m_std = norm.fit(matching_scores)
    nm_mean, nm_std = norm.fit(non_matching_scores)
    print(len(matching_scores),len(non_matching_scores))
    matching_x = np.linspace(np.min(matching_scores), np.max(matching_scores), 1000)
    non_matching_x = np.linspace(np.min(non_matching_scores), np.max(non_matching_scores), 1000)
    matching_dist = norm.pdf(matching_x, m_mean, m_std)
    non_matching_dist = norm.pdf(non_matching_x, nm_mean, nm_std)
    # The below data is used to create,
    np.save(f"results_other/matching_gaussian{version}.npy", matching_dist)
    np.save(f"results_other/non_matching_gaussian{version}.npy", non_matching_dist)
    
    #here we are calculating the average of both standard devantion means to find threshold 1
    th_mean = (nm_mean+m_mean)/2
    #here we are finding the second threshold
    th_inter = solve(m_mean,nm_mean,m_std,nm_std)
    # saving results_other in a txt file
    with open(f"results_other/threshold_{version}.txt", "w+") as file:
        file.write(f"{th_inter} {th_mean}\n")
    return th_inter, th_mean 


if __name__ == "__main__":
    '''change the directory containing the dataset below
    '''
    data_dir = "../student_resampled_wav/" # the path to the directory which contains our vector files
    version = 1
    if len(sys.argv) > 1: version = int(sys.argv[1])
    get_threshold(data_dir, version)
