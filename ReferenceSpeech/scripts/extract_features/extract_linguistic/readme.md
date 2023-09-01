## How to extract the linguistic features
 
### Step 1: Generate ASR transcripts

In this work, we used whisper to generate transcriptions, particularly the
scripts by Blair Johnson, at:
https://github.com/Blair-Johnson/batch-whisper/tree/main


### Step 2: Prepare the environemnts for linguistic feature extraction - part 1

We have created a separate environment for the extraction of BlaBla fetaures.

For the BlaBla environment, run:
```
conda create -n blabla python==3.8.13
pip install -r requirements/blablaenv_requirements.txt
```
Then follow the steps in https://github.com/novoic/blabla to install BlaBla. 
We have installed BlaBla from source.

Before running the script to extract linguistic features, run the following 
command on the terminal:
``` 
export CORENLP_HOME='$HOME/path/to/corenlp_v452'
```

### Step 3: Extract linguistic features - part 1

Uptade the paths in rows 177 and 178 of main() of extract_linguistic.py.
Run:
```
conda activate blabla
python extract_linguistic.py
deactivate
```

### Step 4: Prepare the environemnts for linguistic feature extraction - part 2 (ambiguous pronouns)

We have created a separate environment for the extraction of reference chains.

Download the repository in https://github.com/vdobrovolskii/wl-coref, and
follow the steps in the Preparation section of the readme file, and the point 1 
in the Evaluation section.

Run the following to install extra packages necessary for computing the features:
```
conda activate wl-coref
pip install -r requirements/wlcorefenv_extra_requirements.txt
```

Copy the following directories/files from the repository directory into this directory:
* directory coref/
* directory data/
* file config.toml

If you encounter a *segmentation fault*, you can use the following steps to 
create the conda environment instead: 

```
conda create --name wl-coref --file requirements/wlcoref_conda_spec_file.txt
conda activate wl-coref
pip install -r  requirements/wlcoref_pip_spec_file.txt
```


### Step 5: Extract linguistic features - part 2 (ambiguous pronouns)
Update the paths in rows 179 and 180 of main() in ambiguous_coref.py.
Then, run:
```
conda activate wl-coref
python ambiguous_coref.py
deactivate
```