import pandas as pd
import numpy as np

SEED=741
np.random.seed(seed=SEED)

from sklearn.preprocessing import StandardScaler
from scipy import stats

from utils.utils import subselect_metada


# functions to test whether two features come from the same distribution (null 
# hypothesis), or from different distribution (alternative hypothesis)

def mann_whitney_u(x, y, alpha=0.01, effect_size=True):
    u1, pvalue = stats.mannwhitneyu(
        x, y, alternative='two-sided', nan_policy="omit")
    
    significant_difference = pvalue <= alpha

    if effect_size:
        # get lengths without nans
        nx, ny = np.sum(~np.isnan(x)), np.sum(~np.isnan(y))
        
        # get u2
        u2 = nx*ny - u1

        # get z
        u = min(u1, u2)
        n = nx + ny
        z = (u - nx*ny/2 + 0.5) / np.sqrt(nx*ny * (n + 1)/ 12)

        # get effect size r
        r = z / np.sqrt(n)

    return pvalue, significant_difference, r


def df_significant_difference(popul1, popul2, feature_lst, friendly_summary=True):
    pvalues = []
    significance = []
    effect_size = []
    for f in feature_lst:
        pvalue, significant, r = mann_whitney_u(
            popul1[f].values, popul2[f].values)
        pvalues.append(pvalue)
        significance.append(significant)
        effect_size.append(r)

    df = pd.DataFrame({
        "features": feature_lst, "pvalues": pvalues,
         "significant": significance, 
         "effect_size": effect_size})

    if friendly_summary:
        significant_df = df[df.significant == True]
        print (
            len(significant_df), "out of", len(df), 
            "features that show a significant difference between the 2 populations")
        print (significant_df.sort_values(by=['effect_size']))

    return df


#def bad_feats_in_reading(meta, feats_df, gender, min_age, max_age, global_scale=False, separate_scale=False):
def feats_failling_mwut(
    meta, feats_df, feature_names, g1_criteria, g2_criteria,
    cols_for_data_selection=None, global_scale=False, separate_scale=False):

    # subselect metadata according to cols_for_data_selection
    if cols_for_data_selection is not None:
        df = subselect_metada(meta, cols_for_data_selection)
    else:
        df = meta.copy()

    # subselect features according to metadata df
    feats_ = feats_df[feats_df.wav_file_id.isin(df.wav_file_id.values)]
    
    # scale
    if global_scale:
        scaler = StandardScaler()  # TODO: allow use of other scalers.
        f_vals = scaler.fit_transform(feats_[feature_names])
        feats = pd.DataFrame(f_vals, columns=feature_names)
        feats["wav_file_id"] = feats_["wav_file_id"].values
    else:
        feats = feats_
    
    g1_files = subselect_metada(df, g1_criteria).wav_file_id.values
    g2_files = subselect_metada(df, g2_criteria).wav_file_id.values
    
    # get g1 feats and scale them separately:
    g1_feats = feats[feats.wav_file_id.isin(g1_files)]
    if separate_scale:
        scaler = StandardScaler()
        g1_vals = scaler.fit_transform(g1_feats[feature_names])
        g1 = pd.DataFrame(g1_vals, columns=feature_names)
        g1["wav_file_id"] = g1_feats["wav_file_id"].values
    else:
        g1 = g1_feats
    
    # get g2 feats and scale them separately:
    g2_feats = feats[feats.wav_file_id.isin(g2_files)]
    if separate_scale:
        scaler = StandardScaler()
        g2_vals = scaler.fit_transform(g2_feats[feature_names])
        g2 = pd.DataFrame(g2_vals, columns=feature_names)
        g2["wav_file_id"] = g2_feats["wav_file_id"].values
    else:
        g2=g2_feats

    results_df = df_significant_difference(g1, g2, feature_names, friendly_summary=True)
    feats_from_diff_populations = results_df[results_df.significant].features.values
    
    return feats_from_diff_populations