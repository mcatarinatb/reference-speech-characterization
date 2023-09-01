import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet

import matplotlib.pyplot as plt

class OutlierRemoval:
    """Class to remove outliers. Currently, the supported methods
    for outlier removal are:
    - IQR - interquantile range. This is a univariate approach
    - outlier removal with iteraive pca - It is a multivariate 
    approach. requires manual definition of the thresholds for 
    outlier dedinition, for each of the 2 PCA components. 
    Interesting to explore the data.
    - Mahalanobis distance using the Minimum Covariance Determinant
    (MCD). This is a multivariate approach.
    Allows ploting the PCA of data, before and after outlier removal.
    """
    def __init__(self, data_df, id_field_name="wav_file_id", feature_columns=None, column_to_color=None):
        # int
        self.data_df = data_df.reset_index(drop=True)
        self.id_field_name = id_field_name
        self.feature_columns = feature_columns
        self.outlier_ids = []
        self.column_to_color = column_to_color
        if (column_to_color is not None) and (column_to_color not in data_df.columns):
            print ("WARNING:", column_to_color, "is not provided in data_df. pca will not be colored accordingly.")
            self.column_to_color = None
        
        # get data from df to x array
        self._get_data_from_df()


    def pca(self, pca_outlier_limits, plot_pca_after_removal=True):
        assert np.array(pca_outlier_limits).shape[0] == 2
        assert np.array(pca_outlier_limits).shape[1] == 2
        
        # run PCA
        if self.column_to_color is not None:
            colored_attribute = self.data_df[self.column_to_color].values
        else:
            colored_attribute = None
        x_transformed = fit_plot_pca(self.x, colored_attribute=colored_attribute, return_tranform=True)

        # identify outliers if limits for components are provided
        pca_outlier_limits = np.array(pca_outlier_limits)
        outliers_indexes = [i 
                            for i, data in enumerate(x_transformed) 
                            if np.any(
                                ((data[0]<pca_outlier_limits[0,0]) or (data[0]>pca_outlier_limits[0,1])) 
                                or 
                                ((data[1]<pca_outlier_limits[1,0]) or (data[1]>pca_outlier_limits[1,1]))
                            )]

        # printing outlier limits:
        print("pca limits:", pca_outlier_limits)

        # exclude outliers
        self._exclude_outliers(outliers_indexes)
        
        if plot_pca_after_removal:
            print ("PCA after outlier removal")
            if self.column_to_color is not None: # we need to re-so this because outliers 
                                             # were removed, and data_df changed 
                colored_attribute = self.data_df[self.column_to_color].values
            fit_plot_pca(self.x, colored_attribute=colored_attribute)
            
    def iterative_pca(self, multi_pca_outlier_limits):
        assert len(np.array(multi_pca_outlier_limits).shape) == 3
        
        for i, limit in enumerate(multi_pca_outlier_limits):
            print ("Iteration", i+1, "of iterative PCA")
            if i+1 < len(multi_pca_outlier_limits):
                self.pca(pca_outlier_limits=limit, plot_pca_after_removal=False)
            else:
                self.pca(pca_outlier_limits=limit, plot_pca_after_removal=True)
            
    def mcd_mahalanobis_dist(self):
        # Mahalanobis distance using the Minimum Covariance Determinant (MCD), proposed in [1], which is
        # a robust estimator of covariance, and should be less affected by outliers.
        # Based on: https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html#id2
        # [1]: P. J. Rousseeuw. Least median of squares regression. J. Am Stat Ass, 79:871, 1984.

        # scale x
        #t = PCA(n_components=2)
        #self.x = t.fit_transform(self.x)

        
        # fit MCD and get distances
        print ()
        print ("shape of x used to determine MinCovDet (should be n_samples, n_features)", self.x.shape)
        print ()
        robust_cov = MinCovDet(support_fraction=1).fit(self.x)
        # support fraction should be between 0 and 1. Defaults to None, 
        # that means: n_sample + n_features + 1) / 2 (??)
        # however, fixing 1 makes the outlier removal deterministic.

        # cubic root of mahalanobis dist, for outlier detection
        robust_mahal_dist = robust_cov.mahalanobis(self.x - robust_cov.location_) ** (1./3)

        # compute IQR to get cuttoft threshold:
        outlier_idxs = iqr_univariate_outliers(np.array([robust_mahal_dist]).T)

        # generate report
        print ()
        print ("**** Outlier removal based on MCD - Mahalanobis dist: report ****")
        print ("BEFORE OUTLIER REMOVAL:")
        print ("boxplot of mahalanobis dist of original population:")    
        plt.boxplot([robust_mahal_dist]) #, widths=0.25)
        plt.show()
        print ("pca of original population:")
        if self.column_to_color is not None:
            colored_attribute = self.data_df[self.column_to_color].values
        else:
            #colored_attribute = None
            colored_attribute = ["outlier" if i in outlier_idxs else "inlier" for i in range(len(self.x))]
        fit_plot_pca(self.x, colored_attribute=colored_attribute)
        
        # exclude outliers
        self._exclude_outliers(outlier_idxs)
        
        print ("boxplot of mahalanobis dist after outlier removal:")    
        plt.boxplot([data for i,data in enumerate(robust_mahal_dist) if i not in outlier_idxs]) #, widths=0.25)
        plt.show()
        print ("pca after outlier removal:")
        if self.column_to_color is not None: # we need to re-so this because outliers 
                                             # were removed, and data_df changed 
            colored_attribute = self.data_df[self.column_to_color].values
        else:
            colored_attribute = None
        fit_plot_pca(self.x, colored_attribute=colored_attribute)
        
    def iqr(self, show_before_after_pca=True):
        # compute IQR
        outlier_idxs = iqr_univariate_outliers(self.x)
        
        print ("**** Outlier removal based on IQR: report ****")
        print ("pca of original population:")
        if self.column_to_color is not None:
            colored_attribute = self.data_df[self.column_to_color].values
        else:
            colored_attribute = None
        fit_plot_pca(self.x, colored_attribute=colored_attribute)
        
        # exclude outliers
        self._exclude_outliers(outlier_idxs)
        
        print ("pca after outlier removal:")
        if self.column_to_color is not None: # we need to re-so this because outliers 
                                             # were removed, and data_df changed 
            colored_attribute = self.data_df[self.column_to_color].values
        fit_plot_pca(self.x, colored_attribute=colored_attribute)
    
    def _get_data_from_df(self):
        # get feature values and category names
        if self.feature_columns is not None:
            self.x = self.data_df[self.feature_columns].values
        else:
            self.x = self.data_df.drop(columns=[self.id_field_name]).values
        
        # Normalizing data here was making the covariance matrix not full rank
        #scaler = MinMaxScaler() #StandardScaler() #MinMaxScaler()
        #scaler = StandardScaler() #MinMaxScaler()
        #self.x = scaler.fit_transform(self.x)

    
    def _exclude_outliers(self, outlier_idxs):
        if len(outlier_idxs):
            self.outlier_ids.extend(self.data_df.loc[outlier_idxs][self.id_field_name].values)

            print ("outliers removed (", len(outlier_idxs), "/", len(self.data_df), "): - first 5 listed below")
            print (self.data_df.loc[outlier_idxs][self.id_field_name].values[:5])

            # update data_df and x
            self.data_df = self.data_df.drop(outlier_idxs)
            self.data_df = self.data_df.reset_index(drop=True)
            self._get_data_from_df()
                
        else: #6
            print ("No outliers detected")

        
        
# functions that support the outlier removal class:
def fit_plot_pca(x, colored_attribute=None, return_tranform=False, nam_friendly=False):
    """
    colored_attribute is a list or array with len = len(x). 
    It can be, for example, gender/age/origin_dataset
    """
    # run pca
    if nam_friendly:
        raise NotImplemented
        # _, _, _, x_transformed, Ye = ppca(x,d=2,dia=False)
        # pca implementation that allows missing values
        # print (len(x), len(Ye), len(x_transformed))
    else:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        pca = PCA(n_components=2)
        x_transformed = pca.fit_transform(x)
    
    # plot
    if colored_attribute is None:
        plt.scatter(x_transformed[:,0], x_transformed[:,1], color="blue", label="inlier", alpha=0.5, s=10)
    else:
        colored_attribute = np.array(colored_attribute)
        unique_vals, count = np.unique(colored_attribute, return_counts=True)
        # hack to sort unique vals form largest occurence, 
        # to allow better visualization in plot
        count_sort_ind = np.argsort(-count)
        unique_vals = unique_vals[count_sort_ind]
        # func for generating random colors
        #cmap = get_cmap(len(unique_vals))
        #for i,val in enumerate(unique_vals):
        #    # temporary hack to get nice labels in paper
        #    if val == "clac_healthy":
        #        label = "CLAC"
        #    elif val == "voxceleb_annotated_usa_concatenated":
        #        label = "VoxCeleb"
        #    elif val == "timit_concatenated":
        #        label = "TIMIT"
        #    else:
        #        label = val
        #    # get elements in x_transformed that have colored_attribute=val
        #    xi = x_transformed[colored_attribute == val]            
        #    plt.scatter(xi[:,0], xi[:,1], color=cmap(i), label=label, alpha=0.2, s=10)
        #    #plt.show()
        
        # temporary, to fix the colors
        colors = ["blue", "orange"]
        for i, val in enumerate(unique_vals):
            # get elements in x_transformed that have colored_attribute=val
            xi = x_transformed[colored_attribute == val]            
            plt.scatter(xi[:,0], xi[:,1], color=colors[i], label=val, alpha=0.5, s=10)
    plt.legend()
    plt.show()
    
    if return_tranform:
        return x_transformed
    
def iqr_univariate_outliers(x):
    """Computes IQR. Discards points:
            > Q3 + 1.5 IQR
            < Q1 - 1.5 IQR
    """
    Q1 = np.quantile(x, 0.25, axis=0)
    Q3 = np.quantile(x, 0.75, axis=0)
    IQR = Q3 - Q1
    not_iqr_outlier_bool = ((x > (Q1 - 1.5 * IQR)) & (x < (Q3 + 1.5 * IQR))).all(axis=1)
    #not_iqr_outlier = x[not_iqr_outlier_bool]
    outlier_idxs = np.where(not_iqr_outlier_bool == False)[0] 

    return outlier_idxs #, not_iqr_outlier  

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n+1)
    