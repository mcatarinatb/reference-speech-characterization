"""
This script extracts praat features, by calling praat_features.py.
It requires 
 - a .csv file with metada data information, namely on column 
 containing the paths to the wav files.
 or
 - a directory where all wav files are stored.
This scripts accepts a list of datasets to process sequentially.
Paths in the beginning of main() may need to be updated.
"""
 

import sys
import os
import pandas as pd
import glob
from tqdm import tqdm
from multiprocessing import Pool
import json
import argparse
import subprocess

from praat_features import praat_audio_reader
from praat_features import pitch_hnr_jitter_shimmer, measureFormants, speech_rate
 

def files_in_dir(path, ext=".wav"):
    ext_chars = len(ext)
    files_list = list()
    for (dirpath, _, filenames) in os.walk(path):
        files_list += [os.path.join(dirpath, file) for file in filenames if file[-ext_chars:] == ext]
    return files_list

def append_record_to_json(record, filepath):
    with open(filepath, 'a') as f:
        json.dump(record, f)
        f.write("\n")

def load_json_file(filepath):
    data = [] 
    with open(filepath) as f:
        for line in f: 
            try: 
                data.append(json.loads(line)) 
            except ValueError: 
                print ("Skipping invalid line {0}".format(repr(line))) 

    return data

def compute_feats_for_1_wav(wave_file, json_output="feats.json"):
    print ("processing wav file", wave_file)
    sound = praat_audio_reader(wave_file)
    feats_dict = {"file": wave_file}
                    
    # if sound returned is None no features are extracted.
    if sound is not None:
        # compute F0, shimmer, jitter, hnr
        p_hnr_jit_shim_dict = pitch_hnr_jitter_shimmer(sound)
        feats_dict.update(p_hnr_jit_shim_dict)

        # compute formants
        formants_dict = measureFormants(sound)
        feats_dict.update(formants_dict)

        # compute speech rate features
        speech_rate_dict = speech_rate(sound, minpause=0.2)
        feats_dict.update(speech_rate_dict)

    # write dict to file
    append_record_to_json(feats_dict, json_output)


def main(): 

    root = "/path/to/ReferenceSpeech/"
    data_info_dir = root + "data_info/"
    features_dir = root + "features/"
    datasets_metadata = ["dementiabank"]
    multiprocess = True 
    n_threads = 40
    overwrite_feats = True
    get_wavs_from = "metadata_file" # options: {"wav_dir", "metadata_file"}
    

    for d in datasets_metadata:
        print (d)
        # define dataset specific paths
        metadata_path = data_info_dir + d + ".csv"
        csv_feats_path = features_dir + "/" + d + "/praat_feats.csv"
        json_feats_path = features_dir + "/" + d + "/praat_feats.json" 

        # delete features file if overwrite
        subprocess.call(["mkdir", "-p", features_dir + "/" + d + "/" ]) 
        if overwrite_feats:
            if os.path.exists(json_feats_path):
                os.remove(json_feats_path)

        # get wav list
        print ("Listing all audio files...")
        if get_wavs_from == "wav_dir":
            # list all audio files in dir:
            files_list = files_in_dir(wav_dir, ext=ext)
            print (len(files_list), ext, " files found.")
        elif get_wavs_from == "metadata_file":
            meta_df = pd.read_csv(metadata_path)
            files_list = meta_df.wav_path.values
        else:
            raise NotImplementedError

        # Go through all the wave files in the folder and measure all the acoustics
        if multiprocess:
            print("Starting feature extraction using multiprocessing...")
            pool = Pool(processes=n_threads)
            for wave_file in files_list:
                pool.apply_async(
                    compute_feats_for_1_wav, 
                    (wave_file, json_feats_path)
                    )
            pool.close()
            pool.join()
        
        else:
            print("Starting feature extraction...")
            for wave_file in tqdm(files_list):
                compute_feats_for_1_wav(wave_file, json_feats_path) 

        # read json
        feats_dict = load_json_file(json_feats_path)

        # convert to pandas dataframe
        df = pd.DataFrame.from_dict(feats_dict)

        # save to file
        df.to_csv(csv_feats_path, index=False, header=True)
                    


if __name__ == "__main__":
    main()

