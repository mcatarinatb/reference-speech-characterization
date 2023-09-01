from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# scale/normalization functions

def minmax_scale_func(X, test=None, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
    if X is not None:
        X = scaler.fit_transform(X)
    if test is not None:
        test = scaler.transform(test)
    return X, test, scaler

def mean_std_scale_func(X, test=None, scaler=None, mean=0):
    if scaler is None:
        scaler = StandardScaler()
    if X is not None:
        X = scaler.fit_transform(X)
    if test is not None:
        test = scaler.transform(test)
        
    if mean != 0:
        if X is not None:
            X = X + mean # sums mean to all elements, to shif the mean
        if test is not None:
            test = test + mean
    return X, test, scaler

def normalize(data=None, X=None, test=None, scaler=None, train_scaler=True, mode="minmax"):
    """
    X will be used to fit the scaler, and then be transformed by it.
    test will be transformed by the scaler
    data will be interpreted as X if traim_scaler=True, and as test if train_scaler=False.
    """
    if mode is None:
        print ("[INFO]: no scaling was performed. Scaling mode set to None.")
        return X, test, scaler
    
    if train_scaler == True:
        if (data is not None) and (X is None) and (test is None):
            X = data
        elif (X is not None) and data is None:
            pass
        else:
            raise ValueError("train_scaler set to True, but both either both X and data are given, or none.")
    else:        
        if X is not None:
            raise ValueError("train_scaler set to False, but X is not None. Consider passing it as test or data.")
        if (data is not None) and (test is None):
            test = data
        elif (test is not None) and (data is None):
            pass
        else:
            raise ValueError("train_scaler set to False, but both either both test and data are given, or none.")
    
    if mode == "minmax":
        return minmax_scale_func(X, test, scaler)
    elif mode == "mean_0_std_1":
        return mean_std_scale_func(X, test, scaler, mean=0)
    elif mode == "mean_1_std_1":
        return mean_std_scale_func(X, test, scaler, mean=1)
    else:
        raise NotImplementedError 