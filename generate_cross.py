from match_score import *
import numpy as np
from tqdm import tqdm
import os,sys

def generate_match_scores(folder,version=1):
    filenames = list(filter(lambda x:x.split('.')[-1]=='npy', os.listdir(folder)))
    data = np.zeros((len(filenames),len(filenames)),dtype=float)
    print((len(filenames)*len(filenames)))
    count = 0
    for i in range(len(filenames)):
        for j in range(len(filenames)):
            if(i!=j):
                file1data = np.load(os.path.join(folder,filenames[i]))[0]
                file2data = np.load(os.path.join(folder,filenames[j]))[0]
                data[i][j] = match_score(file1data,file2data,version)
    np.save(f"results/matchscores{version}.npy", data)

if __name__=='__main__':
    folder = input("Enter Folder Location : ")
    version = int(input("Enter Version : "))
    generate_match_scores(folder, version)
