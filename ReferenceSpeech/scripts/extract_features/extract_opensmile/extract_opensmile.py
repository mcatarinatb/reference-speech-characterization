"""
This script extracts openSMILE features, eg. eGeMAPSv02.
It requires a .csv file with metada data information. The only requires column 
in this csv file is a columns containing the paths to the wav files.
This scripts accepts a list of datasets to process sequentially.
Update paths in the beginning of main().
"""


import os
import numpy as np
import pandas as pd
import opensmile

def extract(
    metadata_path, feat_set="ComParE_2016", feat_level="functionals", 
    out_path="compare_2016.csv", wav_path_col="wav_path"
    ):
    # read data info
    df = pd.read_csv(metadata_path)
    wav_lst = df[wav_path_col].values

    # initialize smile extractor
    if (feat_set == "ComParE_2016") and (feat_level=="functionals"):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals
            )
    elif (feat_set == "eGeMAPSv02") and (feat_level=="functionals"):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
            )
    else:
        raise NotImplementedError        

    n_iter=len(wav_lst)
    is_first = True
    for i, wav in enumerate(wav_lst):
        print ("[", i+1, "/", n_iter, "]", end="\r")
        if os.path.isfile(wav): 
            if is_first:
                feats = smile.process_file(wav)
                is_first = False
            else:
                feats = feats.append(smile.process_file(wav))
            

    # convert indexes to columns:
    feats = feats.reset_index()
    feats = feats.drop(columns=["start", "end"])

    if not wav_path_col == "wav_path":
        print (
            "Using different paths under column <", 
            wav_path_col, 
            "> for feature extraction. But path recorded is the original,",
            "under column < wav_path >")
        tmp_df = df[[wav_path_col, "wav_path"]]
        tmp_df = tmp_df.rename(columns={wav_path_col: "file"})

        new_feats_df = pd.merge(feats, tmp_df, on="file")
        assert len(feats) == len(new_feats_df)

        new_feats_df=new_feats_df.drop(columns=["file"])
        new_feats_df = new_feats_df.rename(columns={"wav_path": "file"})

        feats = new_feats_df.copy()

    if len(out_path):
        feats.to_csv(out_path, index=False)
    else:
        return feats


def main():
    root = "/path/to/ReferenceSpeech/"
    data_info = root + "data_info/"
    features_dir = root + "features/"
    feat_sets = ["eGeMAPSv02"] #["ComParE_2016", "eGeMAPSv02"]
    datasets_metadata = ["dementiabank"]
    wav_path_col = "wav_path"

    for d in datasets_metadata:
        for f in feat_sets:
            out_dir = features_dir + d 
            os.makedirs(out_dir, exist_ok=True)
            extract(
                data_info + d + ".csv", 
                feat_set=f, 
                feat_level="functionals", 
                out_path=out_dir + "/" + f + ".csv",
                wav_path_col=wav_path_col
                )


if __name__ == '__main__':
    main()
