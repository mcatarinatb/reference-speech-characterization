import pandas as pd
import numpy as np
import random

SEED=741
np.random.seed(seed=SEED)
random.seed(SEED)


class RefIntEstimator:
    def __init__(self, 
                 data_df, feature_columns, id_field_name="wav_file_id", 
                 upper_percentile=97.5, lower_percentile=2.5, seed=SEED):
        
        np.random.seed(seed=seed)
        
        self.data_df = data_df.reset_index(drop=True)
        self.id_field_name = id_field_name
        self.feature_columns = feature_columns
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        
        # get data from df to self.x array
        self._get_data_from_df()
        
        
    def get_ri_with_ci(self, method="np_ri", use_50p_for_ri=False):
        """
        Computes the RI for 1000 datasets, which correpond to resamples
        of the original dataset, via bootstrapping. This provides a 
        confidence intreval for the lower and upper limit of the RI.
        If the width of the CI > 0.2 * width of RI, the RI is considered
        invalid.
        
        If use_50p_for_ri, the RI provided correspond to the 50th percentile
        of the 1000 lower and upper bounds generated via bootstrapping. Else,
        the RI corresponds to the bounds obtained using the original dataset.
        """
        self.compute_ref_interval(method=method)
        
        (
            ci_lb, ci_ub, 
            feats_large_cil, feats_large_ciu,
            lb_50th, up_50th
        ) = self.ci_via_bootsrapping(n_resample=1000, confidence=0.99)
        
        if use_50p_for_ri:
            lower_bounds = lb_50th
            upper_bounds = up_50th
        else:
            [lower_bounds, upper_bounds] = self.ri
        
        # save reference intervals as a dataframe
        df = pd.DataFrame({
            "feature": self.feature_columns,
            "RI_lower_limit": lower_bounds,
            "RI_upper_limit": upper_bounds,
            "CI_for_lower_limit": [(l,u) for l,u in zip(ci_lb[0], ci_lb[1])],
            "CI_for_upper_limit": [(l,u) for l,u in zip(ci_ub[0], ci_ub[1])],
        })
        df["valid_CI_lower_limit"] = True
        df["valid_CI_upper_limit"] = True
        df.loc[df.feature.isin(feats_large_cil), "valid_CI_lower_limit"] = False
        df.loc[df.feature.isin(feats_large_ciu), "valid_CI_upper_limit"] = False
        
        return df

    
    def compute_ref_interval(self, method="np_ri", data=None, return_ri=True):
        update_self_ri = False
        if data is None:
            data = self.x
            update_self_ri = True
        
        if method == "np_ri":
            ri = np_ri(data,
                       lower_percentile=self.lower_percentile, 
                       upper_percentile=self.upper_percentile)
            
        elif method == "parametric":
            ri = parametric_ri(data)
            
        else:
            raise NotImplemented
        
        if update_self_ri:
            self.ri = ri
        
        if return_ri:
            return ri
    
    
    def ci_via_bootsrapping(self, n_resample=1000, confidence=0.99):
        # threshold to define if reference interval is valid, i.e, if
        # confidence interval exceeds 20% of reference interval width
        safety_check = 0.20
        
        if confidence < 1:
            print ("[INFO]: Assuming confidence is given between 0 and 1.")
            confidence = 100*confidence
        else:
            print ("[INFO]: Assuming confidence is given between 0 and 100.")
        
        lower_bounds_lst = []
        upper_bounds_lst = []
        
        # resample data with bootstrapping
        for i in range(n_resample):
            index = np.random.choice(self.x.shape[0], self.x.shape[0], replace=True)
            sampled_data = self.x[index]

            [lower_bounds, upper_bounds] = self.compute_ref_interval(data=sampled_data)
            lower_bounds_lst.append(lower_bounds)
            upper_bounds_lst.append(upper_bounds)

        # compute confidence interval
        lower_bounds_arr = np.stack(lower_bounds_lst)
        upper_bounds_arr = np.stack(upper_bounds_lst)

        ci_lower_bound = np_ri(lower_bounds_arr, 
                               lower_percentile=(100-confidence)/2, 
                               upper_percentile=confidence/2)

        ci_upper_bound = np_ri(upper_bounds_arr, 
                               lower_percentile=(100-confidence)/2, 
                               upper_percentile=confidence/2)
        
        # compute 50th percentile of the ci for each bound:
        lower_bound_50th = np.nanpercentile(lower_bounds_arr, 
                                            50, 
                                            axis=0, 
                                            interpolation="linear")
        upper_bound_50th = np.nanpercentile(upper_bounds_arr, 
                                            50, 
                                            axis=0, 
                                            interpolation="linear")  
        
        # check if confidence interval exceeds 20% of reference interval width
        # get ci widths
        ci_width_lower_bound = ci_lower_bound[1] - ci_lower_bound[0]
        ci_width_upper_bound = ci_upper_bound[1] - ci_upper_bound[0]
        
        # get ri widths
        if not hasattr(self, 'ri'):
            self.compute_ref_interval(return_ri=False)    
        ri_width = self.ri[1] - self.ri[0]
        
        feat_names = np.array(self.feature_columns)
        feats_w_large_ci_lowerb = []
        feats_w_large_ci_upperb = []
        if np.any(ci_width_lower_bound > safety_check*ri_width):
            print ("[WARNING]: The following features have the confidence interval",
                   "on the lower bound larger than 20% of the reference interval")
            feats_w_large_ci_lowerb = feat_names[ci_width_lower_bound > safety_check*ri_width].tolist()
            print (feats_w_large_ci_lowerb)
        else:
            print ("[INFO]: Lower bound confidence intervals verified (<20% of ri) for all features")
        
        if np.any(ci_width_upper_bound > safety_check*ri_width):
            print ("[WARNING]: The following features have the confidence interval",
                   "on the upper bound larger than 20% of the reference interval")
            feats_w_large_ci_upperb = feat_names[ci_width_upper_bound > safety_check*ri_width].tolist()
            print (feats_w_large_ci_upperb)
        else:
            print ("[INFO]: Upper bound confidence intervals verified (<20% of ri) for all features")
        
        return (
            ci_lower_bound, ci_upper_bound, 
            feats_w_large_ci_lowerb, feats_w_large_ci_upperb, 
            lower_bound_50th, upper_bound_50th
        )
    
    def partition_data(self):
        pass
    
    def _get_data_from_df(self):
            self.x = np.array(self.data_df[self.feature_columns].values, dtype=float)
            # dtype float to make sure None are converted to np.nan

            
# functions that support the outlier removal class:
def np_ri(values, lower_percentile=2.5, upper_percentile=97.5, avoid_nans=True):
    # Default numpy method (method='linear')
    if avoid_nans:      
        upper_bound = np.nanpercentile(values, 
                                upper_percentile, 
                                axis=0, 
                                interpolation="linear")
        lower_bound = np.nanpercentile(values, 
                                lower_percentile, 
                                axis=0,
                                interpolation="linear")
    else:
        upper_bound = np.percentile(values, 
                                upper_percentile, 
                                axis=0, 
                                interpolation="linear")
        lower_bound = np.percentile(values, 
                                lower_percentile, 
                                axis=0,
                                interpolation="linear")        
    return [lower_bound, upper_bound]


def parametric_ri(data):
    # mean +- 0.96 std
    # avoids nan values
    # requires data to be normally distributed
    # make sure you use box cox power trasnform in case it is not
    
    # deal with nan values
    non_nan_vals = ~np.isnan(data)
    
    mean = np.mean(data, axis=0, where=non_nan_vals)
    std = np.std(data, axis=0, where=non_nan_vals)
    lower_bound = mean - 0.96*std
    upper_bound = mean + 0.96*std
    return [lower_bound, upper_bound]
