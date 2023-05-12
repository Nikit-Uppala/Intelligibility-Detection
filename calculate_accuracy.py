import pandas as pd
import numpy as np
import os,sys

def CorrectFileName(val):
    parts = val.split('.')[0].split('_')
    if(len(parts)>1):
        splitit = parts[3].split('-')
        if(len(splitit)==1): parts[3]=str(int(parts[3]))
        else: parts[3]=str(int(splitit[0]))+'-'+str(int(splitit[1]))
    return '_'.join(parts)


def ret_dict(statfile, val):
    df = pd.read_csv(statfile).to_numpy()
    fdict = {}
    for i in df:
        fdict[CorrectFileName(i[0])]=True if i[val]==0 else False
    return fdict


def combine_files(folder,outputfileloc,version):
    files_mean = list(filter(lambda x:(x.split('.')[-1]=='csv' and 'mean' in x.split('_') and version == int(x.split('_')[1])),os.listdir(folder)))
    files_inter = list(filter(lambda x:(x.split('.')[-1]=='csv' and 'inter' in x.split('_') and version == int(x.split('_')[1])),os.listdir(folder)))
    arr = np.array([])
    for i in files_mean:
        try:
            df = pd.read_csv(os.path.join('temp',i)).to_numpy()
            if(arr.__len__()<=1): arr = df
            else: arr = np.concatenate((arr,df))
        except: pass
    final = pd.DataFrame(arr)
    final.to_csv(os.path.join(folder,outputfileloc+'_mean.csv'),index=False,header=False)
    arr = np.array([])
    for i in files_inter:
        try:
            df = pd.read_csv(os.path.join(folder,i)).to_numpy()
            if(arr.__len__()<=1): arr = df
            else: arr = np.concatenate((arr,df))
        except: pass
    final = pd.DataFrame(arr)
    final.to_csv(os.path.join(folder,outputfileloc+'_inter.csv'),index=False,header=False)


def check_acc(combfileloc,statfile,val):
    dic = ret_dict(statfile,val)
    df = pd.read_csv(combfileloc).to_numpy()
    corr = 0
    total = 0
    for i in df:
        if(CorrectFileName(i[0]) in dic): 
            corr += (bool(i[1])==bool(int(dic[CorrectFileName(i[0])])))
            total+=1
    return corr/total


print(check_acc('./temp/combined_mean.csv','./data.csv',4))