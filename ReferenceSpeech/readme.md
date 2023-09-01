# Towards reference speech characterization for health applications
This is the code used in the paper [Towards reference speech characterization for health applications](https://www.isca-speech.org/archive/pdfs/interspeech_2023/botelho23_interspeech.pdf), published at [Interspeech 2023](https://www.interspeech2023.org/), Dublin, Ireland, 20-24 August, 2023.


### Setup
1. Download source code from GitHub
  ``` 
  git clone https://github.com/mcatarinatb/reference-speech-characterization.git 
  ```
2. Go to code directory
  ``` 
  cd reference-speech-characterization/ReferenceSpeech 
  ```
3. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual environment
  ```
  conda create --name ref_speech python==3.8.12
  ```  
4. Activate conda environment 
  ```
  conda activate ref_speech 
  ```
5. Install requirements
  ```
  pip install -r requirements_pip_test.txt
  ```

   
### Description
The directory ReferenceSpeech includes the following subdirectories:

* data_info - should include the .csv files with information about the data. Currently it includes example files for the datasets used in the experiments. Notice that not all columns included in the metadata file are necessary.
* scripts - with the code for feature extraction, creating the reference speech model, and running the experiments.
* results - includes the configurations for the experiments used in the paper, as well as the reference intervals obtained for the reference populations (male and female).
* features - currently empty, but should contain the features in case you would like to use this code to run experiments with your own data.


### How to use
If you would like to reproduce the experiments, or build your own reference models using your data, you will have to:
1. Do your own pre-processing of the data (e.g. downsample to 16kHz, remove possible speech from interviewers, etc.).
2. Create the metada files and place them in data_info/. You can find examples in data_info/.
3. Extract necessary features, as explained in the readme.md file in scripts/extract_features/.
4. Run the experiments in the jupyter notebook in scripts/radar_tool/radar_tool.pynb .

If you want to take a look at the results obtained, such as reference intervals derived for all features using the reference population, you can find those under the results/ .