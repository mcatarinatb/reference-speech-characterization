import pickle
import yaml

def save_to_pkl(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        loaded_pkl = pickle.load(f)
    return loaded_pkl

def make_empty_dict(feature_list):
    d = {}
    for f in feature_list:
        d[f] = None
    return d

def load_yaml_config(features_yaml_path):
    return yaml.safe_load(open(features_yaml_path, 'r')) 