import pandas as pd
import numpy as np

SEED=741
np.random.seed(seed=SEED)

from scipy import stats
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt

# utils functions

def all_same(items):
    return all(all(x == items[0]) for x in items)


def add_del_lst(initial_lst=[], add=None, subtract=None):
    """
    initial_lst is a list
    add is a list of lists to add
    subtract is a list of lists to subtract
    """
    lst = initial_lst.copy()
    if add is not None:
        for l in add:
            lst = lst + l
            
    if subtract is not None:
        for l in subtract:
            lst = [element for element in lst if element not in l]
    
    return lst


def subselect_metada(df, cols_for_data_selection):
    """
    df - pandas dataframe of metadata
    cols_for_data_selection - columns to perform the subselection, e.g.:
        {
        "origin_dataset": ["dataset1", "dataset2"],
        "gender": ["female"],
        "age": (50, 1000)
        }
    """
    meta = df.copy()
    for c in cols_for_data_selection.keys():
        if isinstance(cols_for_data_selection[c], list):
            meta = meta[meta[c].isin(cols_for_data_selection[c])]
        elif isinstance(cols_for_data_selection[c], tuple):
            meta = meta[
                (meta[c] >= cols_for_data_selection[c][0]) & 
                (meta[c] <= cols_for_data_selection[c][1])]
        else:
            raise ValueError("cols_for_data_selection must be a dictionary, which values are lists or tuples.")
    return meta



# get standard deviations for two groups, and plot:
def std_between_groups(df1, df2, feature_lst=None):
    if feature_lst is None:
        feature_lst = [c for c in df1 if c != "wav_file_id"]
    
    std_ratio = []
    too_large = []
    for f in feature_lst:
        std1 = np.std(df1[f].dropna().values)
        std2 = np.std(df2[f].dropna().values)
        if np.min([std1, std2]) == 0:
            ratio = None
        else:
            ratio = np.max([std1, std2])/np.min([std1, std2])
            if ratio > 1.5:
                too_large.append(f)
        std_ratio.append(ratio)
    
    # print summary
    print (len(too_large), "/", len(feature_lst), "features showed a ratio>=1.5. i.e.", len(too_large)/len(feature_lst))
    print (too_large)
    
    # plot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(feature_lst,std_ratio)
    plt.axhline(y=1.5, linewidth=1, color='k')
    plt.xticks(rotation='vertical')
    plt.show()
    
    return too_large


def box_cox_transform(feats_df, feature_names=None):
    print ("[INFO]: Transforming data to normal distributions with Box Cox transform.")
    
    if feature_names == None:
        feature_names = [c for c in feats_df.columns if c!="wav_file_id"]
    
    df = feats_df[feature_names].copy()
    lambdas = []
    shifts = []
    for f in feature_names:
        vals = df[f].values
        
        if all(np.isnan(vals)):
            print (f)
        
        # compute a data shit because box cox only accepts positive data
        if np.nanmin(vals) <= 0:
            shift = np.abs(np.nanmin(vals)) + 1
            vals = vals + shift
        else:
            shift = 0
            
        box_cox_vox_vals, bc_lambda = stats.boxcox(vals)
        df[f] = box_cox_vox_vals
        lambdas.append(bc_lambda)
        shifts.append(shift)
        
    return df, lambdas, shifts


def invert_boxcox(x_transformed, bc_lambda, shift):
    return inv_boxcox(x_transformed, bc_lambda) - shift