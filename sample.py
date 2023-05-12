from tqdm import tqdm
import random
import shutil
import os

def remove_dir(path):
    if os.path.exists(path):  
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)

if __name__=='__main__':
    source_folder = './words_npy_real/'
    train_user = ["jeeva", "anju"]
    dest_folder = './shampled_wav4/'
    limit_train = 1000
    limit_test = 1000
    remove_dir(dest_folder)
    os.mkdir(dest_folder) 
    os.mkdir(os.path.join(dest_folder,'train'))
    os.mkdir(os.path.join(dest_folder,'test'))
    # Test
    print("Test")
    train_files = list(filter(lambda x:x.split('_')[0] not in train_user,os.listdir(source_folder)))
    train_files = list(filter(lambda x:x.split('.')[-1]=='npy',train_files))
    chosen_files = random.choices(train_files,k=limit_test)
    for i in tqdm(chosen_files): shutil.copy2(os.path.join(source_folder,i),os.path.join(dest_folder,'test',i))
    # Train
    print("Train")
    test_files = list(filter(lambda x:x.split('_')[0] in train_user,os.listdir(source_folder)))
    test_files = list(filter(lambda x:x.split('.')[-1]=='npy',test_files))
    chosen_files = random.choices(test_files,k=limit_test)
    for i in tqdm(chosen_files): shutil.copy2(os.path.join(source_folder,i),os.path.join(dest_folder,'train',i))
