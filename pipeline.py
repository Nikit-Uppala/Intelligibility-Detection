import pandas as pd
import numpy as np
import glob
import os
import tqdm
import sys
from match_score import match_score
from intelligibility_detection import is_intelligible

# from intelligibility_detection_multiprocess import is_intelligible
#This file needs to be run twice for both the thresholds one by one and the results are written to respective files
#as of now we need to add the threshold manually in the code.( we plan to change it )
#preserve the result file named (venkatt_1.csv) and  (anju_1.csv) before re-running the code for the second threshold as the code will overwrite the files.

def get_results(data_dir, df, test_speakers, test_speaker, threshold, version):
    """Function to find the intelligibity of test speakers, here we have done it for Intonation_L4 (present in previous dataset)

    Args:
        data_dir (string): the path of the folder containing the data
        df (dataframe): intelligibity of the sentences, (intelligibity column of xlsx file)
        test_speakers (list): name of test speakers 
        test_speaker (string): current test speaker, who's intelligibility we want to find.
        threshold (float): the threshold
        version (int): see the match score for more distription

    Returns:
        dict: file name along with the intelligibity values 
    """
    #To run it for all files replace f"{data_dir}/{test_speaker}_Intonation_L4*.npy" with f"{data_dir}/{test_speaker}*.npy"
    #We are currently running it for only intonation L4 files as we computed the threshold for intonation L4 files only.
    test_speaker_files = sorted(glob.glob(f"{data_dir}/{test_speaker}_Intonation_L4*.npy"))#pulls all the files related to the test speaker
    results = []
    for f in tqdm.tqdm(test_speaker_files):
        filename = os.path.basename(f)
        # file name - name_Intonation_L*.npy
        # 2 parts - name of the speaker and word id
        filename = filename.split("_")
        word_id = "_".join(filename[1:])
        same_word_files = glob.glob(f"{data_dir}/*_{word_id}") #pulls all the files related to the word_id from all the speakers
        control_speaker_features = []
        test_speaker_features = []
        for file in same_word_files:
            name = os.path.basename(file).split("_")[0]
            if name in test_speakers:
                test_speaker_features = np.load(file)[0]
            else:
                # if for other speckers the file is intelligible then add it to the control speaker set else not
                if df[os.path.basename(file).split(".")[0].lower()] == 0:
                    control_speaker_features.append(np.load(file)[0])

        results.append(is_intelligible(test_speaker_features, control_speaker_features, threshold, version)) #last 2 agruments are threshold and version ( we need to run the code for both threshold)
        # results.append(is_intelligible(test_speaker_features, control_speaker_features, 35.268212728056746, num_cores, version))
    filenames = [os.path.basename(x) for x in test_speaker_files]
    final_result = {
        "filename": filenames,
        "intelligible": results
    }
    return final_result

def main():
    """This the main function running the whole programme, i.e. data_copy.csv which contains the intelligibity 
    of sentences( given in xlsx file), data_dir contains the path which had the features, the test_speakers are the
    the one whose intelligibity will checked, rest all will be treated as controll speakers, the results are saved in 
    respective .csv files.
    """
    # A csv file which contains filename, not_intelligible
    df = pd.read_csv("data/data_copy.csv", header=None)
    df = df.values.tolist()
    # A dictionary with filename as key and not_intelligible as value
    df = { x[0].strip().lower(): x[1] for x in df }
    # path to the directory which contains our data files
    data_dir = "/media/nayan/z/wav2vec2_implementaions/wav2vec2_features_for_student_dataset/sentencewise_features" 
    # list of speakers not included in the control speaker set (as venkatt's speech is non-intelligible and anju is to test the code) 
    test_speakers = set(["venkatt", "anju"]) 
    # version of match_score being used
    version = 1 
    with open(f"results/threshold_{version}.txt", "r") as file:
        data = file.read().strip().split()
        thresholds = {"inter": float(data[0]), "mean": float(data[1])}
    for threshold_type in thresholds:
        for test_speaker in sorted(test_speakers):
            # results for each speaker are written to respective files.
            result = get_results(data_dir, df, test_speakers, test_speaker, thresholds[threshold_type], version) 
            result_df = pd.DataFrame(result)
            result_df.to_csv(f"results/{test_speaker}_{version}_{threshold_type}.csv", index=False, header=False)


if __name__ == '__main__':
    num_cores = os.cpu_count()
    if len(sys.argv) > 1:
        num_cores = min(num_cores, int(sys.argv[1]))
    main()