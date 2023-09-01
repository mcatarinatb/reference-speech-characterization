## How to extract the necessary features

### Step 1: extract openSMILE features

Updade the paths in rows 77 to 82 of main() in 
extract_opensmile/extract_opensmile.py.
Then, run:
  ``` 
  cd extract_opensmile
  python extract_opensmile.py
  ```

### Step 2: extract praat features

Updade the paths in rows 77 to 82 of main() in 
extract_praat_feats/compute_feats.py.
Then, run:
  ``` 
  cd extract_praat_feats
  python compute_feats.py
  ```

### Step 3: extract linguistic features

Run:
  ``` 
  cd extract_linguitic
  ```
Then, follow the steps described in the readme.md file in that directory.


### Step 4: compile feature sets into a single file

Go the the config/ directory. There you can find one configuration file for each 
dataset to be processesd and one configuration file for the final feature set, 
named compile_featset.json.  
You should update the configuration files for each dataset by providing the 
relevant paths, or create new configuration files. Note that the same structure
should be used. 

The configuration file to define the feature set should stay unaltered. It 
defines which features are used in each speech task, and which features are
used for defining the outliers.

Update lines 98 to 100 of main() function in compile_features.py.

Run:
```
python compile_features.py
```
