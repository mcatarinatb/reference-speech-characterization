import pandas as pd
import numpy as np
import random
import json
import os 

SEED=741
np.random.seed(seed=SEED)
random.seed(SEED)

from scipy import stats
from scipy.stats import zscore
from scipy.special import inv_boxcox

import plotly.graph_objects as go

from utils.outlier_removal import OutlierRemoval       # outlier removal
from utils.reference_intervals import RefIntEstimator  # class for defining erference interval
from utils.io import *                                 # Function for loading data
from utils.scale import *                              # scale/normalization functions
from utils.plots import *                              # function to creatre the radar plot, and other plots
from utils.utils import *



class SpeechModel:
    def __init__(self, config, scaler=None):
        """Config is a dictionary, as exemplified below"""

        self.config = config
        self.boxcox_transformed = False
        
        # cerating output dir and saving a copy of the configuration
        if os.path.exists(config["output_dir"]):
            self.config["output_dir"] = config["output_dir"].rstrip('/') + "_new_version/"
            print ("[WARNING]: output_dir exists. Moving output_dir to", self.config["output_dir"])
        
        os.makedirs(self.config["output_dir"], exist_ok=True)
        save_to_pickle(self.config["output_dir"] + "/config.pkl", self.config)
        
        # load_data. This creates self.meta_df and self.feats_df
        # which are dataframes that contain the features and 
        # metadata of all datasets used, combined into two 
        # dataframes. meta_df, by default has the columns:
        # ["wav_file_id", "age", "sex", "origin_dataset"]
        # feats_df has the columns "wav_file_id" and all 
        # features included in the feature file.
        # It also creates self.feature_names_lst which is
        # a list of the names of the features to 
        print ("[INFO]: Loading data...")
        self.load_data()

        if config["outlier_removal"]:
            print ("[INFO]: Removing outliers...")
            if self.config["outlier_removal_conf"]["from_id_list"]:
                print ("Removing outliers from id list")
                ids = self.config["outlier_removal_conf"]["outlier_id_list"]
                self.feats_df = self.feats_df[~self.feats_df.wav_file_id.isin(ids)]
                self.feats_df = self.feats_df.drop(columns=[c for c in self.feats_df.columns if "outlier" in c])
                self.meta_df["outlier"] = False
                self.meta_df.loc[self.meta_df.wav_file_id.isin(ids), "outlier"] = True
            
            else:
                print ("Removing outliers with", self.config["outlier_removal_conf"]["method"])
                self.remove_outliers(
                    method=self.config["outlier_removal_conf"]["method"],
                    pca_outlier_limits=self.config["outlier_removal_conf"]["pca_oultier_limits"],
                    column_to_color=self.config["outlier_removal_conf"]["attribute_to_color_in_pca"]
                )
            
            if self.config["outlier_removal_conf"]["save_data_after"]:
                self.meta_df.to_csv(self.config["output_dir"] + "/metadata_after_outlier_rem.csv")
                self.feats_df.to_csv(self.config["output_dir"] + "/feats_after_outlier_rem.csv")

        if config["use_feat_subset"] is not None:
            if isinstance(config["use_feat_subset"], list):
                self.feature_names_lst = config["use_feat_subset"]
                cols = ["wav_file_id"] + self.feature_names_lst 
                self.feats_df=self.feats_df[cols]
            else:
                raise ValueError("config['use_feat_subset'] must be None or a list of features") 
                
        if config["scale"]:
            if config["multi_normalize"]:
                self.multi_normalize(config["multi_normalize_conditions"])
            else:
                self.multi_normalized = False
                _ = self.normalize(pretrained_scaler=scaler)
                
            if self.config["save_data_after_normalize"]:
                if hasattr(self, "scaler"):
                    save_scaler([self.scaler], ["scaler"], dir_path=self.config["output_dir"])
                elif hasattr(self, "scaler_lst"):
                    names = ["scaler_" + str(i) for i in range(len(self.scaler_lst))]
                    save_scaler([self.scaler_lst], ["scaler"], dir_path=self.config["output_dir"])

                self.feats_df.to_csv(self.config["output_dir"] + "/feats_after_scaling.csv")    
    
    def load_data(self):
        # load data
        meta_lst = []
        feats_lst = []
        for i, dataset in enumerate(self.config["datasets"]):
            d_meta_df, d_feats_df = load_data(
                dataset=dataset, 
                feat_type=self.config["features"], 
                feature_dir=self.config["feature_dir"], 
                metadata_dir=self.config["metadata_dir"], 
                sex=self.config["sex"], 
                min_age=self.config["age"]["min"], 
                max_age=self.config["age"]["max"],
                cols_for_data_selection=self.config["subselect_data"],
                distinct_feats_for_outlier_detect=self.config["distinct_feats_for_outlier_detect"]
            )
            meta_lst.append(d_meta_df)
            feats_lst.append(d_feats_df)
            #if i:
                #self.meta_df = self.meta_df.append(d_meta_df)
                #self.feats_df = self.feats_df.append(d_feats_df)
                
            #else:
            #    self.meta_df = d_meta_df.copy()
            #    self.feats_df = d_feats_df.copy()
        
        feats_lst = [df.reindex(sorted(df.columns), axis=1) for df in feats_lst]
        assert all_same([m.columns for m in meta_lst]), "columns of meta_lst are not in the same order"
        assert all_same([f.columns for f in feats_lst]), "columns of feats_lst are not in the same order"
        self.meta_df = pd.concat(meta_lst, sort=False) 
        self.feats_df = pd.concat(feats_lst, sort=False)         
                
        # exclude some features, if needed
        if len(self.config["feature_columns_to_drop"]):
            self.feats_df = self.feats_df.drop(columns=self.config["feature_columns_to_drop"])
        
        # store names of features
        feat_names = self.feats_df.drop(columns=["wav_file_id"]).columns
        if self.config["distinct_feats_for_outlier_detect"]:
            self.feature_names_lst = [c for c in feat_names if "_foroutlierdetect" not in c]
            self.feature_for_outlier_detect_lst = [c for c in feat_names if "_foroutlierdetect" in c]
            self.complete_feature_names_lst = feat_names
        else:
            self.feature_names_lst = feat_names
        
                
    def get_data_subset(self, cols_for_data_selection):
        """
        cols_for_data_selection is. adictionary. It's keys correspond
        to column names. Ex: {"diagnosis": ["dementia, depression"], 
        "hamilton_score": (0, 7)}. If value is passed as a tuple, it's 
        interpreted as min and max values.
        """
        meta = self.meta_df.copy()
        feats = self.feats_df.copy()
        
        for c in cols_for_data_selection.keys():
            print ("selecting data based on...", cols_for_data_selection[c])
            if isinstance(cols_for_data_selection[c], list):
                meta = meta[meta[c].isin(cols_for_data_selection[c])]
            elif isinstance(cols_for_data_selection[c], tuple):
                meta = meta[
                    (meta[c] >= cols_for_data_selection[c][0]) & 
                    (meta[c] <= cols_for_data_selection[c][1])]
            else:
                raise ValueError("cols_for_data_selection must be a dictionary, whose values are lists or tuples.")
        
        tmp = pd.merge(meta, feats, on="wav_file_id")
        feats = tmp[feats.columns]
 
        return meta, feats

    
    def normalize(self, pretrained_scaler=None, scaling_mode=None, train_scaler=None, use_all_data=True, feats_df=None): #TODO!!!!! 
        print ("[INFO]: Scaling data...")
        if scaling_mode is None:
            scaling_mode = self.config["scaling_mode"]
        if train_scaler is None:
            train_scaler=self.config["train_scaler"]
        
        if use_all_data:
            feats_df = self.feats_df[self.feature_names_lst].copy()
            wav_ids = self.feats_df.wav_file_id
        elif feats_df is not None:
            wav_ids = feats_df.wav_file_id
            feats_df = feats_df[self.feature_names_lst].copy()
        #elif self.subset_feats is not None:
        #    feat_vals = self.subset_feats[self.feature_names_lst].copy()
        else:
            raise ValueError("With use_all_data=False, pass a feats_df")
        
        x, y, scaler = normalize(data=feats_df, 
                  mode=scaling_mode, 
                  train_scaler=train_scaler, 
                  scaler=pretrained_scaler)
        
        if bool(x is None) == bool(y is None):
            raise ValueError("Either x or y must be None. They can't be both None.")
        elif x is None:
            feats_vals=y
        else:
            feats_vals=x
        
        if use_all_data:
            self.feats_df[self.feature_names_lst] = feats_vals # this may very well not work :o
            self.scaler=scaler
        
        feats_df[self.feature_names_lst] = feats_vals
        feats_df["wav_file_id"] = wav_ids
        return feats_df, scaler
    
    
    def multi_normalize(self, condition_lst):
        """
        condition_lst is a list of dicts. Each dict contains:
        - train_scaler: bool
        - pretrained_scaler: either a scaler (if train_scaler is True) or None (if train scaler is False)
        - cols_for_data_selection: a dictionary. It's keys correspond
        to column names. Ex: {"diagnosis": ["dementia, depression"], 
        "hamilton_score": (0, 7)}. If value is passed as a tuple, it's 
        interpreted as min and max values.
        """
        self.multi_normalized = True
        meta_lst = []
        feats_lst = []
        scaler_lst = []
        for c in condition_lst:
            train_scaler = c["train_scaler"]
            pretrained_scaler = c["pretrained_scaler"]
            cols_for_data_selection = c["cols_for_data_selection"]
            
            # select_data:
            meta_df, feats_df = self.get_data_subset(cols_for_data_selection)
            
            if len(meta_df):
                #normalize:
                feats_df, scaler = self.normalize(
                    pretrained_scaler=pretrained_scaler, 
                    scaling_mode=self.config["scaling_mode"], 
                    train_scaler=train_scaler, 
                    use_all_data=False, feats_df=feats_df)

                meta_lst.append(meta_df)
                feats_lst.append(feats_df)
                scaler_lst.append(scaler)

        # concat:
        assert all_same([m.columns for m in meta_lst]), "columns of meta_lst are not in the same order"
        assert all_same([f.columns for f in feats_lst]), "columns of feats_lst are not in the same order"
        self.meta_df = pd.concat(meta_lst, sort=False) 
        self.feats_df = pd.concat(feats_lst, sort=False) 
        self.scaler_lst = scaler_lst
        self.multi_normalizing_conds = condition_lst
    
    
    def metadata_report(self):
        # TODO!
        pass
    
    
    def remove_outliers(self, method="mcd_mahalanobis_dist", 
                        pca_outlier_limits=[[[-10000000,3000],[-10000000,3000]],
                                            [[-10000000,10000000], [-200, 200]],
                                            [[-10000000,10000000], [-200,200]]],
                       column_to_color=None):
        """
        removes outliers using the class OutlierRemoval.
        Implemented methods include mcd_mahalanobis_dist, 
        iterative_pca, and iqr.
        """
        # prepare df that includes only the
        # features relevant for outlier detection
        if self.config["distinct_feats_for_outlier_detect"]:
            feature_names = self.feature_for_outlier_detect_lst
        else:
            feature_names = self.feature_names_lst
            
        
        # if column to color, include it in data_df
        if column_to_color is not None:
            col2col_df = self.meta_df[["wav_file_id", column_to_color]]
            tmp_feats_df = pd.merge(self.feats_df, col2col_df, on="wav_file_id")
        else:
            tmp_feats_df = self.feats_df.copy()
        
        # remove outliers
        print ()
        print ("number of variables: ", len(feature_names))
        print ("number of observations: ", len(tmp_feats_df))
        outlier_rem = OutlierRemoval(
            tmp_feats_df, id_field_name="wav_file_id", 
            feature_columns=feature_names,
            column_to_color=column_to_color
        )
        print ("method for outlier removal: ", method)
        
        if method == "iqr":
            outlier_rem.iqr()
        elif method == "iterative_pca":
            print ("[WARNING]: ensure that you pass suitable pca_outlier_limits.")
            outlier_rem.iterative_pca(multi_pca_outlier_limits=pca_outlier_limits)
        elif method == "mcd_mahalanobis_dist":
            outlier_rem.mcd_mahalanobis_dist()
        else:
            raise NotImplementedError
        
        # save only relevant features, from here onwards, excluding features for outlier detect
        if self.config["distinct_feats_for_outlier_detect"]:
            non_outlier_ids = outlier_rem.data_df.wav_file_id.values
            new_feats_df = self.feats_df[self.feats_df.wav_file_id.isin(non_outlier_ids)].copy()
            self.feats_df = new_feats_df[self.feature_names_lst + ["wav_file_id"]].copy()
        else:
            self.feats_df = outlier_rem.data_df
            non_outlier_ids = outlier_rem.data_df.wav_file_id.values
        
        self.meta_df["outlier"] = True
        self.meta_df.loc[self.meta_df.wav_file_id.isin(non_outlier_ids), "outlier"] = False
        
        if column_to_color is not None:
            if column_to_color in self.feats_df.columns:
                self.feats_df = self.feats_df.drop(columns=[column_to_color])
    
    
    def get_reference_intervals(self, method="np_ri", 
                                feature_subset=None, data_subset=None, 
                                boxcox_subgroups_lst=None, reverse_boxcox=True, 
                                return_friendly=True):
        """
        return ri_df with columns:
        'feature', 'RI_lower_limit', 'RI_upper_limit', 'CI_for_lower_limit',
       'CI_for_upper_limit', 'valid_CI_lower_limit', 'valid_CI_upper_limit'
        """
        # getting feature list (either self.feature_names_lst or subset)
        feature_names_lst = self._get_feature_names(feature_subset)

        # convert to gaussian using box-cox if method is parametric
        if method == "parametric":
            if data_subset is None:
                print ("[INFO]: Transforming self.feats_df via box cox")
                if boxcox_subgroups_lst is not None:
                    self.boxcox_transform_by_subgroups(boxcox_subgroups_lst)
                else:
                    self.boxcox_transform()
            else:
                print ("[INFO]: Transforming data_subset via box cox")
                if boxcox_subgroups_lst is not None:
                    feats_df = self.boxcox_transform_by_subgroups(boxcox_subgroups_lst, data_subset)
                else:
                    feats_df, _, _ = self.boxcox_transform(data_subset)
                
        if data_subset is None:
            feats_df = self.feats_df.copy()                
        
        # get reference intervals
        ri_estimator = RefIntEstimator(feats_df, feature_names_lst)
        ri_df = ri_estimator.get_ri_with_ci(
            method=method, use_50p_for_ri=self.config["use_50p_for_ri"])
        self.ri_df = ri_df
        
        
        if (method=="parametric") & (reverse_boxcox):
            if data_subset is None:
                ri_df = self.invert_boxcox(ri_df, cols_to_process=["RI_lower_limit", "RI_upper_limit"])
                self.ri_df = ri_df
            else:
                raise ValueError("Can't reverser boxcox if it was not perfomed on the entire feature_df. Do it yourself.")
        
        # return intervals 
        if data_subset is not None:
            print (["[WARNING]: denorming options and frieldy summary not yet available when data_subset is not None."])
            return ri_df, feats_df
        
        elif method == "parametric":
            print (["[WARNING]: denorming options and frieldy summary not yet available for parametric reference interval estivation."])
            return ri_df
        
        elif (not return_friendly) or (self.config["scaling_mode"] is None):
            return ri_df
            
            
        elif self.config["scaling_mode"] == "mean_1_std_1":
            raise NotImplementedError
        else:
            ri_df["valid_interval"] = ri_df["valid_CI_lower_limit"] & ri_df["valid_CI_upper_limit"]
            to_denorm_df = ri_df[['feature','RI_lower_limit', 'RI_upper_limit']].set_index('feature').T
            
            if not self.multi_normalized:
                denormed_vals = self.denorm_values(to_denorm_df)
                ri_df["denormed_RI_lower_limit"] = denormed_vals[0]
                ri_df["denormed_RI_upper_limit"] = denormed_vals[1]
                return ri_df[["feature", 'RI_lower_limit', 'RI_upper_limit', "denormed_RI_lower_limit", "denormed_RI_upper_limit", "valid_interval"]] 
            
            else:
                cols_of_interest = ["feature", 'RI_lower_limit', 'RI_upper_limit', 'valid_interval']
                for i, s in enumerate(self.scaler_lst):
                    denormed_vals = self.denorm_values(to_denorm_df, scaler=s)
                    ri_df["denormed_RI_lower_case" + str(i+1)] = denormed_vals[0]
                    ri_df["denormed_RI_upper_case" + str(i+1)] = denormed_vals[1]
                    cols_of_interest.extend(["denormed_RI_lower_case" + str(i+1), "denormed_RI_upper_case" + str(i+1)])
                return ri_df[cols_of_interest] 
            
            
    def denorm_values(self, to_denorm_df, scaler=None):
        """
        to_denorm_df should be a dataframe with columns 
        corresponding to the features we want to denorm, 
        and the rows correspond to the values.
        """  
        if scaler == None:
            scaler = self.scaler
        
        features_to_denorm = [c for c in to_denorm_df.columns if not c == "features"]
        if (len(features_to_denorm) == len(scaler.feature_names_in_)) and np.all(features_to_denorm ==scaler.feature_names_in_):
            denormed_vals = scaler.inverse_transform(to_denorm_df)

        else: # hack to use pretrained scaler to denorm data that contains only a subset of features
            hack_df = pd.DataFrame(columns=self.feature_names_lst)
            for f in to_denorm_df.columns:
                hack_df[f] = to_denorm_df[f]
            hack_denormed_vals = scaler.inverse_transform(hack_df)
            hack_denormed_df = pd.DataFrame(hack_denormed_vals, columns=self.feature_names_lst)
            denormed_vals = hack_denormed_df[to_denorm_df.columns].values

        return denormed_vals
        
        
    def build_radar_plot(self, plot_mean=True, plot_ri=True, 
                         return_fig=True, save_fig=True,
                         output_img_path="background_img.png",
                         legend_prefix="reference", mean_color="random", 
                         feature_subset=None): 
        
        # getting feature list (either self.feature_names_lst or subset)
        feature_names_lst = self._get_feature_names(feature_subset)
        
        # getting feature values, and ignoring NANs
        feat_vals_df = self.feats_df[feature_names_lst]
        vals_none = feat_vals_df.astype(object).replace(np.nan, "None").values
        non_nan_vals = ~np.equal(vals_none, "None")
        feat_vals = np.array(feat_vals_df.values, dtype=float) # to convert None to np.nan
        values_to_plot = []
        subplot_legend = []
        colors = []
        
        if plot_mean:
            mean = np.mean(feat_vals, axis=0, where=non_nan_vals)
            values_to_plot.append(mean)
            subplot_legend.append(legend_prefix + "_mean")
            if mean_color == "random":
                r = lambda: random.randint(0,255)
                rgb = '#%02X%02X%02X' % (r(),0,r())
                colors.append(rgb)
            else:
                colors.append(mean_color)
            
        if plot_ri:
            if (not hasattr(self, 'ri_df')) or np.any(self.ri_df.feature.values.tolist() != feature_names_lst):
                _ = self.get_reference_intervals(method="np_ri", feature_subset=feature_subset)
            values_to_plot.append(self.ri_df.RI_lower_limit.values)
            subplot_legend.append(legend_prefix + "_RI_lower_limit")
            values_to_plot.append(self.ri_df.RI_upper_limit.values)
            subplot_legend.append(legend_prefix + "_RI_upper_limit")
            colors.extend(["lightgreen", "lightgreen"])
        
        # radar plot:
        fig = radar_plot(
            categories=feature_names_lst, 
            values_to_plot=values_to_plot, 
            sub_plot_legend=subplot_legend,
            colors=colors)
        
        # save and display figure
        if save_fig:
            fig.write_image(output_img_path)
            
        if return_fig:
            return fig
        
        
    def add_trace_to_radar_plot(self, to_plot, fig, original_plot_feature_names, save_fig=True, legend_prefix="test_dataset", line_opacity=1, color=None):
        """
        color only for to_plot== "all"
        """
        
        # getting feature list (either self.feature_names_lst or subset)
        original_plot_feature_names = self._get_feature_names(original_plot_feature_names)
        
        # verify that the features used to build the original radar plot
        # are included in self.feature_names_lst
        assert np.all([f in self.feature_names_lst for f in original_plot_feature_names])
        feat_vals_df = self.feats_df[original_plot_feature_names] #.values
        
        # getting feature values and ignoring NANs (usefull only for to_plot="mean")
        vals_none = feat_vals_df.astype(object).replace(np.nan, "None").values
        non_nan_vals = ~np.equal(vals_none, "None")
        feat_vals = np.array(feat_vals_df.values, dtype=float) # to convert None to np.nan

        # defining what to plot:
        if to_plot == "mean":
            vals_to_plot = np.mean(feat_vals, axis=0, where=non_nan_vals)   
            #print ("values that will be plot are: (fisrt shape, then values)")
            #print (vals_to_plot.shape)
            #print (vals_to_plot)
        elif to_plot == "median":
            vals_to_plot = np.median(feat_vals, axis=0)
        elif to_plot == "random_instance": 
            wav_file_id = self.get_example(
                sex=self.config["instance_sex"], 
                age=self.config["instance_age"]["exact"], 
                min_age=self.config["instance_age"]["min"], 
                max_age=self.config["instance_age"]["max"], 
                cols_for_data_selection=self.config["instance_cols_for_data_selection"])
            vals_to_plot = self.feats_df[self.feats_df.wav_file_id == wav_file_id][original_plot_feature_names].values.squeeze()
            #print (vals_to_plot.shape)
            #print (vals_to_plot)
        
        elif to_plot == "all":
            vals_to_plot = feat_vals
            
        elif to_plot in self.feats_df.wav_file_id.values:
            vals_to_plot = self.feats_df[self.feats_df.wav_file_id == to_plot][original_plot_feature_names].values.squeeze()
        
        else:
            print (str(vals_to_plot) + "is not a valid wav_file_id, nor a valid option.")
            raise NotImplementedError

        # get a random color
        if color is None:
            r = lambda: random.randint(0,255)
            rgb = '#%02X%02X%02X' % (r(),0,r())
        else:
            rgb = color
        
        #plot
        if not to_plot == "all": 
            fig.add_trace(go.Scatterpolar(
                r=vals_to_plot,
                theta=[str(c) for c in np.arange(len(original_plot_feature_names))],
                name=legend_prefix,
                line_color=rgb,
                opacity=line_opacity,
                line_width=1,
            ))
        
        else:
            for val in vals_to_plot:
                fig.add_trace(go.Scatterpolar(
                    r=val,
                    theta=[str(c) for c in np.arange(len(original_plot_feature_names))],
                    #name=legend_prefix,
                    line_color=rgb,
                    opacity=line_opacity, 
                    line_width=1, 
                ))                

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True,
                # todo - find a smart way to define the range
                #range=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] # axis range
                #range=[-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
                range = (-4., 5.)
                )),
            showlegend=False,
            width=1000, 
            height=1000
        )
        fig.show()

        if save_fig:
            fig.write_image(legend_prefix + ".png")
            
        return vals_to_plot
            
            
    def get_example(self, sex=None, age=None, min_age=None, max_age=None, cols_for_data_selection=None):
        meta = self.meta_df.copy() 
        if sex is not None:
            meta = meta[meta.sex == sex]
        if min_age is not None:
            meta = meta[meta.age >= min_age]
        if max_age is not None:
            meta = meta[meta.age <= max_age]
        if age is not None:
            meta = meta[meta.age == age]    
        # get only examples of the selected columns:
        if cols_for_data_selection is not None:
            for c in cols_for_data_selection.keys():
                print (cols_for_data_selection[c])
                if isinstance(cols_for_data_selection[c], list):
                    meta = meta[meta[c].isin(cols_for_data_selection[c])]
                elif isinstance(cols_for_data_selection[c], tuple):
                    meta = meta[
                        (meta[c] >= cols_for_data_selection[c][0]) & 
                        (meta[c] <= cols_for_data_selection[c][1])]
                else:
                    raise ValueError("cols_for_data_selection must be a dictionary, which values are lists or tuples.")

            

        assert len(meta), "The specified combination of sex, age and diagnosis does not yield any example"

        wav_file_id = np.random.choice(meta.wav_file_id.values, 1)

        return wav_file_id[0]

    
    def are_vals_inside_ri(self, vals, vals_feat_names, present_denormed_vals=True):        
        # getting feature list (either self.feature_names_lst or subset)
        vals_feat_names = self._get_feature_names(vals_feat_names)
        
        # verify that the features to verify
        # are included in self.feature_names_lst
        assert np.all([f in self.feature_names_lst for f in vals_feat_names])
        
        if (not hasattr(self, 'ri_df')) or np.any([f not in self.ri_df.feature.values for f in vals_feat_names]):
            _ = self.get_reference_intervals(method="np_ri", feature_subset=vals_feat_names)
        df = self.ri_df.copy()
        
        vals_df = pd.DataFrame({"val_to_check": vals, "feature": vals_feat_names})
        df = pd.merge(df, vals_df, on=["feature"])
        df["in_ref_interval"] = False
        df.loc[
            (df.val_to_check >= df.RI_lower_limit) & (df.val_to_check <= df.RI_upper_limit), 
            "in_ref_interval"
        ] = True
        
        # Denorm vals
        if present_denormed_vals:
            df_to_denorm = pd.DataFrame([vals], columns=vals_feat_names)
            denormed_vals = self.denorm_values(df_to_denorm)
            df["val_to_check_denormed"] = denormed_vals.squeeze()
        
        # Mention how many features do not have info, 
        # and update df to remove features without info
        print ("features without information:", len(df[df["val_to_check"].isna()]))
        df = df[~df["val_to_check"].isna()]
        
        # print features outside the reference interval
        print ("features outside reference values:", len(df[~df["in_ref_interval"]]))
        cols_to_print = [
            "feature", "RI_lower_limit", "RI_upper_limit", 
            "val_to_check", "valid_interval", "in_ref_interval"]
        if present_denormed_vals:
            cols_to_print.extend(["denormed_RI_lower_limit", "denormed_RI_upper_limit", "val_to_check_denormed"])
        print (df[~df["in_ref_interval"]][cols_to_print].round(3))
        
        # check ref interval + confidence interval
        df["in_ri_plus_ci"] = False
        df["RI_lower_limit_minus_ci"] = [ci[0] for ci in df.CI_for_lower_limit.values]
        df["RI_upper_limit_plus_ci"] = [ci[1] for ci in df.CI_for_upper_limit.values]
        df.loc[
            (df.val_to_check >= df.RI_lower_limit_minus_ci) & (df.val_to_check <= df.RI_upper_limit_plus_ci), 
            "in_ri_plus_ci"
        ] = True
        
        # print features outside the reference interval
        print ("features outside reference values + confidenece intervals:", len(df[~df["in_ri_plus_ci"]]))
        print (df[~df["in_ri_plus_ci"]]["feature"].values)
        
        return df
    
    
    def are_individual_feats_in_ri(
        self, feats_df, ri_df=None, feature_names=None, 
        friendly_summary=True, ref_feats=None, 
        mode="inside_ri"):
        
        # assert mode is valid:
        assert mode in ["inside_ri", "below_upper_bound", "over_lower_bound"], "invalid mode selected."

        # getting feature list (either self.feature_names_lst or subset)
        if feature_names is None:
            feature_names = [c for c in feats_df.columns if c != "wav_file_id"]
        else:
            # getting feature list (either self.feature_names_lst or subset)
            feature_names = self._get_feature_names(feature_names)
        
        # verify that the features to verify
        # are included in self.feature_names_lst
        assert np.all([c in self.feature_names_lst for c in feature_names])
        
        # if RIs are not defined yet, define them:
        if ri_df is None:
            if ref_feats is None:
                ref_feats = self.feats_df.copy()
                if (not hasattr(self, 'ri_df')) or np.any([f not in self.ri_df.feature.values for f in feature_names]):
                    _ = self.get_reference_intervals(method="np_ri", feature_subset=feature_names)

            else:
                if (not hasattr(self, 'ri_df')) or np.any([f not in self.ri_df.feature.values for f in feature_names]):
                    _ = self.get_reference_intervals(method="np_ri", feature_subset=feature_names, data_subset=ref_feats)
            
            ri_df = self.ri_df

        n_samples_ref_population = []               
        n_samples_subgroup = []
        n_inside_ri = []
        for feature in feature_names:
            if self.config["use_outer_bound_of_CI"]:
                ri_lower = ri_df[ri_df.feature == feature].CI_for_lower_limit.item()[0]
                ri_upper = ri_df[ri_df.feature == feature].CI_for_upper_limit.item()[1]  
            else:
                ri_lower = ri_df[ri_df.feature == feature].RI_lower_limit.item()
                ri_upper = ri_df[ri_df.feature == feature].RI_upper_limit.item()
                       
            v = feats_df[feature].dropna().values
            n_samples_subgroup.append(len(v))
            if mode == "inside_ri":
                n_inside_ri.append(len(v[(v>= ri_lower) & (v<=ri_upper)]))
            elif mode == "below_upper_bound":
                n_inside_ri.append(len(v[v<=ri_upper]))
            elif mode == "over_lower_bound":
                n_inside_ri.append(len(v[v>= ri_lower]))
            
            n_samples_ref_population.append(len(ref_feats[feature].dropna()))
            
        n_samples_ref_population = np.array(n_samples_ref_population)
        n_samples_subgroup = np.array(n_samples_subgroup)
        n_inside_ri = np.array(n_inside_ri)
        n_outside_ri = n_samples_subgroup - n_inside_ri
        summary_df = pd.DataFrame({
            "feature": feature_names,
            "n_non_nan_samples": n_samples_subgroup,
            "n_non_nan_samples_in_ref": n_samples_ref_population,
            "n_samples_in_ri": n_inside_ri,
            "prop_in_ri_over_subgroup": [i/n  if n>0 else None for (i, n) in zip(n_inside_ri, n_samples_subgroup)],
            "prop_in_ri_over_ref": [i/n  if n>0 else None for (i, n) in zip(n_inside_ri, n_samples_ref_population)],
            "n_samples_out_ri": n_outside_ri,
            "prop_out_ri_over_subgroup": [o/n  if n>0 else None for (o, n) in zip(n_outside_ri, n_samples_subgroup)],
            "prop_out_ri_over_ref": [o/n  if n>0 else None for (o, n) in zip(n_outside_ri, n_samples_ref_population)],
        })
        
        # print summary
        if friendly_summary:
            pass
            # TODO: revisit what would actually be useful in this frindly summary!
            empty_features = summary_df[summary_df.n_non_nan_samples == 0].feature.values
            good_features = summary_df[summary_df.prop_in_ri>=0.959].feature.values
            bad_features = summary_df[summary_df.prop_in_ri<0.959].feature.values
            print (len(empty_features), "empty features:", empty_features)
            print(
                len(good_features), 
                "features with at least 95.9% of population in ri: ",
                good_features
            )
            print(
                len(bad_features), 
                "features with less than 95.9% of population in ri: ",
                bad_features
            )           

        return summary_df
    
    
    def count_feats_out_ri(self, feats_df, ri_df=None, feature_names=None):        
        # getting feature list (either self.feature_names_lst or subset)
        if feature_names is None:
            feature_names = [c for c in feats_df.columns if c != "wav_file_id"]
        else:
            # getting feature list (either self.feature_names_lst or subset)
            feature_names = self._get_feature_names(feature_names)

        # verify that the features to verify
        # are included in self.feature_names_lst
        assert np.all([c in self.feature_names_lst for c in feature_names])

        # if RIs are not defined yet, define them:
        if ri_df is None:
            ref_feats = self.feats_df.copy()
            if (not hasattr(self, 'ri_df')) or np.any([f not in self.ri_df.feature.values for f in feature_names]):
                _ = self.get_reference_intervals(method="np_ri", feature_subset=feature_names)
            ri_df = self.ri_df

        pop_n_feats_in = []
        pop_n_feats_out = []
        pop_n_feats_nan = []
        pop_n_feats_below_lower_limit = []
        pop_n_feats_over_upper_limit = []
        for file in feats_df.wav_file_id:
            feats_in = 0
            feats_out = 0
            feats_nan = 0
            feats_below_lower = 0
            feats_over_upper = 0
            for feature in feature_names:
                v = feats_df[feats_df.wav_file_id == file][feature].item()
                if np.isnan(v):
                    feats_nan +=1
                else:
                    # get limits
                    if self.config["use_outer_bound_of_CI"]:
                        ri_lower = ri_df[ri_df.feature == feature].CI_for_lower_limit.item()[0]
                        ri_upper = ri_df[ri_df.feature == feature].CI_for_upper_limit.item()[1]
                    else:
                        ri_lower = ri_df[ri_df.feature == feature].RI_lower_limit.item()
                        ri_upper = ri_df[ri_df.feature == feature].RI_upper_limit.item()
                    
                    if v <= ri_lower:
                        feats_below_lower += 1
                    if v >= ri_upper:
                        feats_over_upper +=1
                        
                    if (v>= ri_lower) & (v<=ri_upper):
                        feats_in += 1
                    else:
                        feats_out += 1
            pop_n_feats_in.append(feats_in)
            pop_n_feats_out.append(feats_out)
            pop_n_feats_nan.append(feats_nan)
            pop_n_feats_below_lower_limit.append(feats_below_lower)
            pop_n_feats_over_upper_limit.append(feats_over_upper)


        summary_df = pd.DataFrame({
            "n_feats_in": pop_n_feats_in,
            "n_feats_out": pop_n_feats_out,
            "n_feats_nan": pop_n_feats_nan,
            "n_feats_over_upperlimit": pop_n_feats_over_upper_limit,
            "n_feats_below_lowerlimit": pop_n_feats_below_lower_limit
            })
        return summary_df
   
    
    def adjust_ri(self, ri_df_, tolerance=0.05, store_in_self=False):
        """
        receives a dataframe with reference intervals from another
        speech model, and adjusts it to be coherent with the population
        in the current speech model.
        According to https://acutecaretesting.org/en/articles/
        reference-intervals-2--some-practical-considerations
        to validate a RI, no more than 10% of the new population should
        be out of the RI. We will do this separately for each limit 
        of the RI (lower and upper bound). Thus, in our approach, to 
        validate each limite, no more than 5% of the population should
        be above (or below) the upper (or lower) limit.
        """
        ri_df = ri_df_.copy()
        
        # verify that the features to verify
        # are included in self.feature_names_lst
        feature_names = ri_df.feature.values
        assert np.all([c in self.feature_names_lst for c in feature_names])
        
        # Do the thing
        n_samples = []
        n_above_upper_limit = []
        n_below_lower_limit = []
        for feature in feature_names:
            # TODO: consider if it would make sense to use the confidence interval
            ri_lower = ri_df[ri_df.feature == feature].RI_lower_limit.item()
            ri_upper = ri_df[ri_df.feature == feature].RI_upper_limit.item()
                       
            v = self.feats_df[feature].dropna().values
            n_samples.append(len(v))
            
            # over the upper limit and under the lower limir:
            if len(v) > 0:
                n_above_upper_limit.append(len(v[v>ri_upper])/len(v))
                n_below_lower_limit.append(len(v[v<ri_lower])/len(v))
            else:
                n_above_upper_limit.append(None)
                n_below_lower_limit.append(None)
                
        df = pd.DataFrame({
            "feature": feature_names,
            "n_non_nan_samples": n_samples,
            "prop_over_ri": n_above_upper_limit,
            "prop_under_ri": n_below_lower_limit
        })
        
        # merge df with the dataframe containing ri info
        df = pd.merge(df, ri_df, on="feature")
        
        df.loc[df.prop_over_ri > tolerance, "RI_upper_limit"] = float('inf')        
        df.loc[df.prop_under_ri > tolerance, "RI_lower_limit"] = float('-inf')
        
        minus_inf_series = pd.Series([(float('-inf'), float('-inf'))] * len(df))
        inf_series = pd.Series([(float('inf'), float('inf'))] * len(df))
        df.loc[df.prop_over_ri > tolerance, "CI_for_upper_limit"] = minus_inf_series
        df.loc[df.prop_under_ri > tolerance, "CI_for_lower_limit"] = inf_series
        
        df["keep_feature"] = True
        df.loc[(df.RI_lower_limit == float('-inf')) & (df.RI_upper_limit == float('inf')), "keep_feature"] = False
        df['keep_feature'] = df['keep_feature'].where(df['RI_lower_limit'].notna(), False)
        df['keep_feature'] = df['keep_feature'].where(df['RI_upper_limit'].notna(), False)
 
        if store_in_self:
            self.ri_df = df[df.keep_feature]
        return df

   
    
    
    def _get_feature_names(self, feature_subset):
        """
        auxiliary function to allow presenting the results only 
        for a subset of the features. This subset can be passed
        as a list, csv file (header only considered, and file
        identifier excluded if on of the following is used: 
        {"wav_path", "file", "file_path", "wav_file_id"}), or 
        json (feature list must be in field "features").
        """
        if feature_subset is None:
            return self.feature_names_lst
        
        elif type(feature_subset) is list:
            assert np.all([f in self.feature_names_lst for f in feature_subset]), "features requested must belong to the original datset files passed to the speech model."
            return feature_subset
        
        elif type(feature_subset) is str:
            if feature_subset.split(".")[-1] == "csv":
                df = pd.read_csv(feature_subset)
                file_identifiers = ["wav_path", "file", "file_path", "wav_file_id"]
                return [f for f in df.columns if f not in file_identifiers]
            
            elif feature_subset.split(".")[-1] == "json":
                data = json.load(open(feature_subset))
                #return data["medium_set_features"]
                return data["medium_set_features_pict"]
                #return data["survived_db_controls"]
            else:
                raise ValueError('Feature subset was passed as a string, that is nor a csv file nor a json.')
     
    
    
    def transform_feats_df_by_task(self, task_dict, drop_features=[]): 
        maptasks={
            "read_speech": ["read_speech"],
            "spont_speech": ["pic_description", "concat_interview_segm"],
            "spont_speech_interview": ["concat_interview_segm"],
            "spont_speech_picture": ["pic_description"],
            "vowel_a": ["vowel_a"]
        }
        mapprefix={
            "read_speech": "RS",
            "spont_speech": "SS",
            "spont_speech_interview": "IS",
            "spont_speech_picture": "Pic",
            "vowel_a": "A"
            ""
        }
        
        for i, key in enumerate(task_dict.keys()):
            # get files
            file_ids = self.meta_df[self.meta_df.task_type.isin(maptasks[key])].wav_file_id.values
            
            # feature names:
            feat_names = task_dict[key]
            feat_names = [f for f in feat_names if f in self.feats_df.columns.values]
            new_feat_names = {}
            for f in feat_names:
                new_feat_names[f] = mapprefix[key] + "#" + f
                
            # get features
            sub_feats_df = self.feats_df[self.feats_df.wav_file_id.isin(file_ids)].copy()
            sorted_file_ids = sub_feats_df.wav_file_id.values
            sub_feats_df = sub_feats_df[feat_names]
            sub_feats_df["wav_file_id"] = sorted_file_ids
            sub_feats_df = sub_feats_df.rename(columns=new_feat_names)

            # merge
            if i:
                feats_df = feats_df.merge(sub_feats_df, how="outer", on=["wav_file_id"])
            else:
                feats_df = sub_feats_df.copy()
            
        if len(drop_features):
            feats_df = feats_df.drop(columns=drop_features, errors='ignore')
        self.feats_df = feats_df
        self.feature_names_lst = [ c for c in feats_df.columns if c != "wav_file_id"]
        return feats_df
    
    
    def boxcox_transform(self, in_feats_df=None):
        
        if in_feats_df is None:
            feats_df, bc_lambdas, shifts = box_cox_transform(
                self.feats_df) #, feature_names=self.feature_names_lst)
            feats_df["wav_file_id"] = self.feats_df.wav_file_id.values
            
            self.boxcox_transformed = True
            self.feats_df = feats_df.copy()
            self.boxcox_lambda = bc_lambdas
            self.boxcox_shift = shifts
        
        else:
            feats_df, bc_lambda, shift = box_cox_transform(in_feats_df)
            feats_df["wav_file_id"] = in_feats_df.wav_file_id.values
            return feats_df, bc_lambda, shift
  

    def boxcox_transform_by_subgroups(self, boxcox_subgroups_lst, data_subset=None):
        transformed_feats = []
        for cols_for_data_selection in boxcox_subgroups_lst:
            sub_meta, sub_feats = self.get_data_subset(cols_for_data_selection)
            
            if data_subset is not None:
                sub_file_lst = sub_meta.wav_file_id.values
                sub_feats = data_subset[data_subset.wav_file_id.isin(sub_file_lst)]
            
            feats, _, _ = box_cox_transform(sub_feats)
            feats["wav_file_id"] = sub_feats.wav_file_id.values
            transformed_feats.append(feats)

        transformed_feats = [df.reindex(sorted(df.columns), axis=1) for df in transformed_feats]
        assert all_same([f.columns for f in transformed_feats]), "columns of feats_lst are not in the same order"
        final_feats = pd.concat(transformed_feats, sort=False) 

        if data_subset is None:
            assert all_same([final_feats.wav_file_id.values, self.feats_df.wav_file_id]), "boxcox_subgroups_lst do not include all data."
            self.feats_df = final_feats
        
        else:
            assert all_same([np.sort(final_feats.wav_file_id.values), np.sort(data_subset.wav_file_id.values)]), "boxcox_subgroups_lst do not include all data."
            return final_feats
        
        
    def invert_boxcox(self, df, cols_to_process):
        df_invert = df.copy()
        for c in cols_to_process:
            invert_vals = []
            assert len(self.boxcox_lambda) == len(self.boxcox_shift)
            assert len(self.boxcox_shift) == len(self.feature_names_lst)
            for feat, bc_lambda, shift in zip(self.feature_names_lst, self.boxcox_lambda, self.boxcox_shift):
                vals = df[df.feature == feat][c].values
                inv_val = invert_boxcox(vals, self.boxcox_lambda, self.boxcox_shift)
                invert_vals.append(inv_val)
            df_invert[c] = invert_vals
        
        return df_invert
        