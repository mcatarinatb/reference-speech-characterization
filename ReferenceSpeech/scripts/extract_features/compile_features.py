"""
This script compiles the distinct feature sets into a single feature file.
It requires:
- configuration file for the features to be included in the final feature set.
This file specifies the features, the source feature set, and the tasks for 
which such features will be considered. See the example in the config folder.
- configuration file for each dataset to be processes, including the tasks in 
each dataset, and the relevant paths. See examples in config folder.

Specify these configuration files, and datasets to be processes in lines 98 
to 100 of main() function.
"""


import pandas as pd
import numpy as np
import json

def load_json_config(path):
    f = open(path)
    data = json.load(f)
    return data


def compile_feat_set(feats_config, data_config):

    feature_groups = []
    files_to_exclude = []
    for group_config in feats_config["details"]:
        # load features
        feats_df, files_to_exclude_feat = load_feats(group_config["source"], data_config)
        files_to_exclude.extend(files_to_exclude_feat)
        if feats_df is not None:
            feats_to_keep = feats_df[group_config["features"]]

            # rename features, if necessary
            if not group_config["rename"] == "none":
                feats_to_keep.columns = [c + group_config["rename"] for c in feats_to_keep.columns]

            # store file name
            feats_to_keep = pd.concat([feats_to_keep, feats_df[["file"]]], axis=1)

            # subselect tasks, if necessary
            if not group_config["datasets_tasks"] == "all":
                include_files = []
                for task_type in group_config["datasets_tasks"]:
                    if task_type in data_config["tasks"].keys():
                        if data_config["tasks"][task_type] == "all":
                            include_files.extend(feats_to_keep["file"].values)
                        else:
                            for task_name in data_config["tasks"][task_type]:
                                task_files = [f for f in feats_to_keep["file"].values if task_name in f]
                                include_files.extend(task_files)
                feats_to_keep = feats_to_keep[feats_to_keep.file.isin(include_files)]

            # add fetaure group to list
            feature_groups.append(feats_to_keep)
        else:
            columns = group_config["features"] + ["file"]
            empty_df = pd.DataFrame(columns=columns)
            feature_groups.append(empty_df)

    # make final dataframe
    df = feature_groups[0]
    for f_df in feature_groups[1:]:
        df = pd.merge(df, f_df, on="file", how="outer")

    # exclude files that are all "all-nan" for any feature set
    print("df initial length:", len(df))
    files_to_exclude = np.unique(files_to_exclude)
    df = df[~df.file.isin(files_to_exclude)]
    print("df length after excluding all-nans:", len(df))
    
    # make some tests
    assert len(df.columns) == 1 + np.sum([len(group["features"]) for group in feats_config["details"]])
    
    return df


def load_feats(ftype, data_config):
    fpath_name = "fpath_" + ftype 
    files_to_exclude = []
    if not data_config[fpath_name] == "none":
        feats_df = pd.read_csv(data_config[fpath_name])

        # if there is a row with all nan's, discard that row
        d_nan = feats_df.isna()
        cols_to_ignore = ['transcript_path', 'file']
        d_nan[cols_to_ignore] = True
        files_to_exclude = feats_df.loc[d_nan.all(axis="columns")].file.values
    else:
        feats_df=None
    return feats_df, files_to_exclude



def main():
    # PATHS
    datasets = ["dementiabank"] 
    feature_config_path="config/compile_featset.json" 
    outpath_key_in_data_config = "fpath_featset"

    for d in datasets:
        print (d)
        dataset_config_path="config/" + d + ".json"
        # load configs
        feats_config = load_json_config(feature_config_path)
        data_config = load_json_config(dataset_config_path)

        # compile features
        df = compile_feat_set(feats_config, data_config)

        # save to file
        df.to_csv(data_config[outpath_key_in_data_config], index=False)


if __name__ ==  '__main__':
    main()