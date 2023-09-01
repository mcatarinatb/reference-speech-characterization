"""
WARNING -- before running this script, first run:
export CORENLP_HOME='$HOME/toolkits/corenlp_v452'

This script extracts linguistic features: 
- blabla features listed in the config file config/features.yaml
Following https://github.com/novoic/blabla
- polarity
- number of repeats
- coherence of adjacent sentences, by measuring similarity. Following 
https://www.sbert.net/docs/quickstart.html
- usage of 1st person pronouns

The usage of ambigugous pronouns is processed in a different script.

This script requires a .csv file with metada data information, namely the 
columns wav_path, and transcript_path.
This scripts accepts a list of datasets to process sequentially.
Paths in the beginning of main() may need to be updates.
"""

# TODO: if too many files in transcription, re-do this to avoid having too many
#       transcriptions in memory - -batch processing
# TODO: merge with scrpt for measuring the usage of ambiguous pronouns

# Notes:
# average word per sentence (using whisper transcription and punctuation), the 
# avg number of words in clac-picnic was 13, and in clac-cookie-theft was 15.
# Research suggest that when sentences have 14 words, readers understand over 
# 90 per cent of information [Wylie(2009), American Press Institute (API) study]
# Thus, we perform sentence segmentation for coherence measures using 14 tokens.

import glob
import yaml
import os
import pandas as pd
import numpy as np
from blabla.document_processor import DocumentProcessor
from tqdm.autonotebook import tqdm
import string
from textblob import TextBlob
import stanza

from utils.coherence import text_segmenter_n_tokens
from utils.coherence import SemanticSimilarityScorer

from utils.utils import make_empty_dict
from utils.utils import load_yaml_config

STANZA_CONFIG='config/stanza_config.yaml'

def extract_linguistic_feats_single_file(text_path, feature_list, lang='en'):

    with DocumentProcessor(STANZA_CONFIG, lang) as doc_proc:
        content = open(text_path).read()
        doc = doc_proc.analyze(content, 'string')
        sentence_list = doc_proc.break_text_into_sentences(content, force_split=False)

    res = doc.compute_features(feature_list)
    print(res)
    return res


def extract_linguistic_feats_many_files(fpaths, file_ids, feature_list, 
                                        lang='en', save_to_file='', n_tokens=14):
    
    print ('Loading transcripts...')
    with DocumentProcessor(STANZA_CONFIG, lang) as doc_proc:
        docs = []
        contents = []
        sentences_lst = []
        for text_path in tqdm(fpaths):
            content = open(text_path).read()
            contents.append(content)
            docs.append(doc_proc.analyze(content, 'string'))
            sentences_lst.append(
                doc_proc.break_text_into_sentences(content, force_split=False))

    #initialize relevant models...
    similarity_model = SemanticSimilarityScorer()
    
    print ('Computing features...')
    res = []
    for doc, content, sentences in tqdm(zip(docs, contents, sentences_lst)):
        if len (content):
            # compute blabla
            feats = doc.compute_features(feature_list)

            # compute polarity
            feats['polarity'] = extract_polarity(content)

            # compute repeats
            repeat_count, repeat_ratio = extract_repeats(content)
            feats['repeat_count'] = repeat_count
            feats['repeat_ratio'] = repeat_ratio

            # compute coherence_sentence
            try:
                _, mean_sim, sim_variability = similarity_model.generate_features(
                    sentences)
            except:
                print (sentences)
                print (content)
                assert 1==0
            feats['mean_coher_sentence'] = mean_sim
            feats['variability_coher_sentence'] = sim_variability

            # compute coherence_n_token
            n_tokens_list = text_segmenter_n_tokens(content, n_tokens=n_tokens)
            _, mean_sim, sim_variability = similarity_model.generate_features(
                n_tokens_list)
            feats['mean_coher_' + str(n_tokens) + 'tokens'] = mean_sim
            feats['variability_coher_' + str(n_tokens) + 'tokens'] = sim_variability

            # count 1st person pronouns
            first_pers_pron_count, first_pers_pron_ratio = count_occorrences(content)
            feats["1st_pronouns"] = first_pers_pron_count
            feats["1st_pronouns_ratio"] = first_pers_pron_ratio

            res.append(feats)
        else:
            feat_names = feature_list.copy()
            feat_names.extend([
                'polarity', 'repeat_count', 'repeat_ratio', 
                'mean_coher_sentence', 'variability_coher_sentence', 
                'mean_coher_' + str(n_tokens) + 'tokens',
                'variability_coher_' + str(n_tokens) + 'tokens',
                ])
            res.append(make_empty_dict(feat_names))
    
    df = pd.DataFrame(res)
    df['transcript_path'] = fpaths
    df['file'] = file_ids

    if len(save_to_file):
        df.to_csv(save_to_file, index=False)
    return df 


def extract_polarity(text):
    return TextBlob(text).sentiment.polarity


def extract_repeats(text):
    # remove punctuation, capital letters, and split into words
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    newlist = words.copy()

    # count repeats
    repeat=0
    for word in words:
        newlist.remove(word)
        if word in newlist:
            repeat=repeat+1
    
    return repeat, repeat/len(words)


def count_occorrences(text, lst = ["i", "me", "mine", "my"]):
    # remove punctuation, capital letters, and split into words
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()

    # make sure element is in lower case
    lst = [elem.lower() for elem in lst]

    counts = [words.count(elem) for elem in lst]
    return sum(counts), sum(counts)/len(words)


def main():
    features_yaml_path='config/features.yaml'

    root = '/path/to/ReferenceSpeech/'
    dataset = 'dementiabank' #'voxceleb_annotated_usa_concatenated' #'clac_healthy' #'timit_concatenated' 
    metadata_path = root + 'data_info/' + dataset + '.csv'
    features_out_dir =  root + '/features/' + dataset + "_v3/"

    # mk dir
    os.makedirs(features_out_dir, exist_ok=True)

    # load feature config
    feat_conf = load_yaml_config(features_yaml_path)

    # get feature list, avoiding unavailable feats:
    featlist = feat_conf['features']

    # get file paths:
    print ('Listing text files...')
    metadf = pd.read_csv(metadata_path)
    fpaths = metadf.transcript_path.values
    file_ids = metadf.wav_path.values

    # extract features
    print ('Starting feature extraction...')
    feats = extract_linguistic_feats_many_files(
        fpaths, file_ids, featlist, feat_conf['language'], 
        save_to_file=os.path.join(features_out_dir, 'linguistic.csv'))

if __name__ == '__main__':
    main()
