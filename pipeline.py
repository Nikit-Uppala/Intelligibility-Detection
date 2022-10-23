import pandas as pd
import numpy as np
import glob
import os
import tqdm
import sys
from match_score import match_score
from intelligibility_detection import is_intelligible
# from intelligibility_detection_multiprocess import is_intelligible


def get_results(data_dir, df, test_speakers, test_speaker, version):
    test_speaker_files = sorted(glob.glob(f"{data_dir}/{test_speaker}*.npy"))
    results = []
    for f in tqdm.tqdm(test_speaker_files):
        filename = os.path.basename(f)
        # file name - name_Intonation_L*.npy
        # 2 parts - name of the speaker and word id
        filename = filename.split("_")
        word_id = "_".join(filename[1:])
        same_word_files = glob.glob(f"{data_dir}/*_{word_id}")
        control_speaker_features = []
        test_speaker_features = []
        for file in same_word_files:
            name = os.path.basename(file).split("_")[0]
            if name in test_speakers:
                test_speaker_features = np.load(file)[0]
            else:
                if df[os.path.basename(file).split(".")[0].lower()] == 0:
                    control_speaker_features.append(np.load(file)[0])
        results.append(is_intelligible(test_speaker_features, control_speaker_features, 35.268212728056746, version))
        # results.append(is_intelligible(test_speaker_features, control_speaker_features, 35.268212728056746, num_cores, version))
    filenames = [os.path.basename(x) for x in test_speaker_files]
    final_result = {
        "filename": filenames,
        "intelligible": results
    }
    return final_result


def main():
    df = pd.read_csv("data/data_copy.csv", header=None) # a csv file which contains filename, not_intelligible
    df = df.values.tolist()
    df = {
        x[0].strip().lower(): x[1] for x in df
    }
    data_dir = "data/sentencewise_features" # path to the directory which contains our data files
    test_speakers = set(["venkatt", "anju"]) # list of speakers not included in the control speaker set
    version = 1 # version of match_score being used
    for test_speaker in sorted(test_speakers):
        result = get_results(data_dir, df, test_speakers, test_speaker, version) # results for each speaker are written to respective files.
        result_df = pd.DataFrame(result)
        result_df.to_csv(f"results/{test_speaker}_{version}.csv", index=False, header=False)
        

if __name__ == '__main__':
    num_cores = os.cpu_count()
    if len(sys.argv) > 1:
        num_cores = min(num_cores, int(sys.argv[1]))
    main()