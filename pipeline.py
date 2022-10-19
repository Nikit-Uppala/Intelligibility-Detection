import pandas as pd
import numpy as np
import glob
import os
import tqdm
from match_score import match_score
from intelligibility_detection import is_intelligible


def main():
    data_dir = "data/sentencewise_features"
    test_speaker = "venkatt"
    version = 1
    test_speaker_files = sorted(glob.glob(f"{data_dir}/{test_speaker}_Intonation_*.npy"))
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
            if name == test_speaker:
                test_speaker_features = np.load(file)[0]
            else:
                control_speaker_features.append(np.load(file)[0])
        results.append(is_intelligible(test_speaker_features, control_speaker_features, 35.268212728056746, version))
    final_result = {
        "filename": test_speaker_files,
        "intelligible": results
    }
    df = pd.DataFrame(final_result)
    df.to_csv(f"results/{test_speaker}_{version}.csv")
        


if __name__ == '__main__':
    main()