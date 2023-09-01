import pandas as pd
import joblib
import json
import pickle
import os 


def load_data(dataset, feat_type, feature_dir, metadata_dir, 
              sex="both", min_age=None, max_age=None, cols_for_data_selection=None,
             final_meta_columns=["wav_file_id", "spk", "age", "gender", "origin_dataset", "task", "task_type", "transcript_diag", "hamilton"],
             distinct_feats_for_outlier_detect=False):
    """
    Loads metadata (from meatdata_dir) and feature values (feature_dir)
    associated with a single dataset, and a single feature set. Allows selecting 
    data based on gender, age and any other columns of the meta_df, that can be 
    passed as a dictionary in cols_for_data_selection.
    
    args:
    : dataset: dataset name.
    : feat_type: currently allows egemaps, egemaps_wo_mfccs, compare, praat
    : feature_dir
    : metadata_dir
    : sex: {'both','male','female'}
    : min_age
    : max_age
    : cols_for_data_selection. Ex: {"diagnosis": ["dementia, depression"], 
            "hamilton_score": (0, 7)}. If value must be a tuple or a list. If
            it is passed as a tuple, it's interpreted as min and max values.
    : final_meta_columns: the columns to keep in the metadata dataframe generated
    : distinct_feats_for_outlier_detect
    """
    
    # get paths
    if (feat_type == "egemaps") or (feat_type == "eGeMAPS"):
        file_name = "eGeMAPSv02.csv"
    elif (feat_type == "compare") or (feat_type == "ComParE"):
        file_name = "ComParE_2016.csv"
    elif (feat_type == "egemaps_wo_mfccs") or (feat_type == "eGeMAPS_wo_mfccs"):
        file_name = "eGeMAPSv02_wo_mfccs.csv"
    elif (feat_type == "praat"):
        file_name = "praat_feats.csv"
    elif (feat_type == "debug"):
        file_name = "debug.csv"
    elif (feat_type == "large_set"):
        file_name = "final_featureset.csv"
    else:
        raise NotImplementedError()
          
    features_path = feature_dir + "/" + dataset + "/" + file_name
    metadata_path = metadata_dir + "/" + dataset + ".csv"
          
    # load metadata:
    meta = pd.read_csv(metadata_path)
    meta["origin_dataset"] = dataset
    
    # get spk_id
    if ("spk" not in meta.columns) and ("spk_id" in meta.columns):
        meta = meta.rename(columns={"spk_id": "spk"})
    
    # get only examples of the selected sex
    if sex != "both":
        if (sex == "female") or (sex == "f"):
            meta = meta[meta.gender=="female"]
        elif (sex == "male") or (sex == "m"):
            meta = meta[meta.gender=="male"]
        else:
            raise ValueError("Plase select a valid option for the field <sex>. Accepted options are {both,male,female}")
    else:
        meta = meta[meta.gender.isin(["male", "female"])] # this excludes "other"
        
        
    # get only examples of the selected age
    if (min_age is not None) and (max_age is not None):
        meta = meta[(meta.age >= min_age) & (meta.age <= max_age)]
    elif bool(min_age is None) != bool(max_age is None):  #xor
        raise ValueError("Either both min_age and max_age are None, or both should be integer numbers.")
    
    
    # get only examples of the selected columns:
    if cols_for_data_selection is not None:
        for c in cols_for_data_selection.keys():
            if isinstance(cols_for_data_selection[c], list):
                meta = meta[meta[c].isin(cols_for_data_selection[c])]
            elif isinstance(cols_for_data_selection[c], tuple):
                meta = meta[
                    (meta[c] >= cols_for_data_selection[c][0]) & 
                    (meta[c] <= cols_for_data_selection[c][1])]
            else:
                raise ValueError("cols_for_data_selection must be a dictionary, which values are lists or tuples.")
    
    # load features:
    feats_df = pd.read_csv(features_path)
    
    # the following is only necessary because some of the csv files 
    # have different field identifiers, which does not allow merging
    for c in ["wav_path", "file", "file_path"]: #"wav_path_after_vad"
        if c in feats_df.columns:
            feats_df = feats_df.rename(columns={c: "wav_file_id"})
        if c in meta.columns:
            meta = meta.rename(columns={c: "wav_file_id"})
            
            
    # separate feats for outlier detection only
    if distinct_feats_for_outlier_detect:
        cols_for_outlier = [c for c in feats_df.columns if "_foroutlierdetect" in c] + ["wav_file_id"]
        feats_for_outlier = feats_df[cols_for_outlier].copy()
        feats_for_outlier = feats_for_outlier.dropna()
        # include in feats_df only those without None values in feats for outlier detection
        feats_df = feats_df[feats_df.wav_file_id.isin(feats_for_outlier.wav_file_id.values)]
        assert feats_df[cols_for_outlier].equals(feats_for_outlier)
    else:
        feats_df = feats_df.dropna()  # remove rows with nan values       
            
    # merge:
    df = pd.merge(meta, feats_df, on="wav_file_id")
    
    # final metadata:
    for c in final_meta_columns:
        if c not in df.columns:
            df[c] = None
    final_meta = df[final_meta_columns]
    
    # final features 
    final_features = df[feats_df.columns]
    
    return final_meta, final_features


def save_scaler(scaler_lst, scaler_names, dir_path=""):
    os.makedirs(dir_path, exist_ok=True)
    for scaler, name in zip(scaler_lst, scaler_names):
        joblib.dump(scaler, dir_path + "/"+ name + '.gz')

        
def load_scaler(path):
    scaler = joblib.load(path)
    return scaler


def save_dict_to_json(path, data_dict):
    with open(path, 'w') as f:
        json.dump(data_dict, f)

        
def save_to_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def export_ri_for_report(
    df1, df2, feature_order=None, 
    path="summary_for_report.csv"):
    
    # assert df1 and df2 are in the same order
    df = pd.merge(
        df1[["feature", "denormed_RI_lower_limit", "denormed_RI_upper_limit"]],
        df2[[
            "feature", "denormed_RI_lower_case1", "denormed_RI_upper_case1", 
            "denormed_RI_lower_case2", "denormed_RI_upper_case2", 
            "denormed_RI_lower_case3", "denormed_RI_upper_case3", 
            "denormed_RI_lower_case4", "denormed_RI_upper_case4"]],
        on="feature"
    )
    
    if feature_order is not None:
        df_order = pd.DataFrame({
            "feature": feature_order,
            "to_order": range(len(feature_order))
        })
        df = df.merge(df, df_order, on="feature")
        df = df.sort_values(by=["to_order"])
    
    # round numbers
    df = df.round(decimals=4)
    
    # get ref intervals:
    males_under50 = ["[" + str(l) + ", " + str(u) + "]" for l, u in zip(df.denormed_RI_lower_case1.values, df.denormed_RI_upper_case2.values)]
    females_under50 = ["[" + str(l) + ", " + str(u) + "]" for l, u in zip(df.denormed_RI_lower_case2.values, df.denormed_RI_upper_case2.values)]
    males_over50 = ["[" + str(l) + ", " + str(u) + "]" for l, u in zip(df.denormed_RI_lower_case3.values, df.denormed_RI_upper_case3.values)]
    females_over50 = ["[" + str(l) + ", " + str(u) + "]" for l, u in zip(df.denormed_RI_lower_case4.values, df.denormed_RI_upper_case4.values)]
    
    all_age_gender = ["[" + str(l) + ", " + str(u) + "]" for l, u in zip(df.denormed_RI_lower_limit.values, df.denormed_RI_upper_limit.values)]
    
    final_df = pd.DataFrame({
        "feature": df1.feature.values,
        "males_under50": males_under50,
        "females_under50": females_under50,
        "males_over50": males_over50,
        "females_over50": females_over50,
        "all_age_gender": all_age_gender
    })
    
    final_df.to_csv(path, index=False)