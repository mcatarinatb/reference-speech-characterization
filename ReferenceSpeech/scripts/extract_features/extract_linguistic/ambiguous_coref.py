"""
WARNING 
-- before running this script, first run:
export CORENLP_HOME='$HOME/toolkits/corenlp_v452'
-- this script runs using conda environment "wl-coref"

This script explores the usage of ambiguous pronouns. To do that:

1. convert text to jsonlines
2. extract reference chains (*)
3. when a reference chain list starts with a 3rd person pronoun (he, she, 
they, etc..), it is considered an ambiguous pronouns (**)

(*) reference chains obatined from: https://github.com/vdobrovolskii/wl-coref
# which was the state of the art (Feb 9 or earlier) in 
# http://nlpprogress.com/english/coreference_resolution.html

(**) method for ambiguous word count from [Iter et al., 2018, Automatic 
Detection of Incoherent speech for diagnosing Schizophrenia]

This script requires a .csv file with metada data information, namely the 
columns wav_path, and transcript_path.
This scripts accepts a single dataset to process each time.
Paths in the beginning of main() may need to be updates.
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import stanza

from tqdm.autonotebook import tqdm

from coref import CorefModel
from coref.tokenizer_customization import *

from utils.utils import make_empty_dict

class CorefAnalysis:
    """
    Adapted from https://github.com/vdobrovolskii/wl-coref predict.py
    """
    def __init__(self, experiment="roberta", config_file="config.toml", 
                 batch_size=None, weights_path=None):
        """
        config_file, default="config.toml"
        batch_size, Adjust to override the config value if you're 
        experiencing out-of-memory issues"
        weights, Path to file with weights to load. If not supplied, in the 
        latest weights of the experiment will be loaded; if there aren't 
        any, an error is raised."
        """
        self._initialize_model(config_file, experiment, batch_size, weights_path)
    
    def compute_ambiguous_refs(self, doc_dict):
        doc_dict = self.extract_reference_chain(doc_dict)
        (ambiguous_ref_count, ambiguous_ref_ratio, ref_chain_count, 
        ref_chain_ratio) = self.count_ambiguous_references(doc_dict)
        return ambiguous_ref_count, ambiguous_ref_ratio, ref_chain_count, ref_chain_ratio

    def extract_reference_chain(self, doc_dict):
        with torch.no_grad():
            doc_dict = self._build_doc(doc_dict)
            result = self.model.run(doc_dict)
            doc_dict["span_clusters"] = result.span_clusters
            doc_dict["word_clusters"] = result.word_clusters

            for key in ("word2subword", "subwords", "word_id", "head2span"):
                del doc_dict[key]
        return doc_dict

    def count_ambiguous_references(self, ref_dict, 
        third_person_pronoun=["he", "his", "him", "himself", 
                            "she", "her", "hers", "helself",
                            "its", "itself", "they", "their", 
                            "theirs", "them", "themselves"]):
        """
        Itentionaly excluded 'it' from the list, because it will appear 
        frequently outside the context of ambiguous refernciation. Such as:
        "It is raining outside."
        """
        words = ref_dict["cased_words"]
        ref_chains = ref_dict["word_clusters"]
        ambiguous_pron = 0
        if len(ref_chains) == 0:
            return 0, 0, 0, 0
        else:
            for r in ref_chains:
                if words[r[0]].lower() in third_person_pronoun:
                    ambiguous_pron +=1   
            return (
                ambiguous_pron, ambiguous_pron/len(ref_chains), 
                len(ref_chains), len(ref_chains)/len(words))
            

    def _initialize_model(self, config_file, experiment, batch_size, weights_path):
        self.model = CorefModel(config_file, experiment)

        if batch_size is not None:
            self.model.config.a_scoring_batch_size = batch_size
        self.model.load_weights(path=weights_path, map_location="cpu",
                        ignore={"bert_optimizer", "general_optimizer",
                                "bert_scheduler", "general_scheduler"})
        self.model.training = False


    def _build_doc(self, doc):
        model = self.model
        filter_func = TOKENIZER_FILTERS.get(model.config.bert_model,
                                            lambda _: True)
        token_map = TOKENIZER_MAPS.get(model.config.bert_model, {})
        word2subword = []
        subwords = []
        word_id = []
        for i, word in enumerate(doc["cased_words"]):
            tokenized_word = (token_map[word]
                            if word in token_map
                            else model.tokenizer.tokenize(word))
            tokenized_word = list(filter(filter_func, tokenized_word))
            word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
            subwords.extend(tokenized_word)
            word_id.extend([i] * len(tokenized_word))
        doc["word2subword"] = word2subword
        doc["subwords"] = subwords
        doc["word_id"] = word_id
        doc["head2span"] = []
        if "speaker" not in doc:
            doc["speaker"] = ["_" for _ in doc["cased_words"]]
        doc["word_clusters"] = []
        doc["span_clusters"] = []

        return doc            


def convert_jsonlines(sent_lst, file_id="", genre_identifier="tc"):
    """
    Example of jsonlines required format:
        {"document_id": "tc_mydoc_001",
         "cased_words": ["Hi", "!", "Bye", "."],
         "sent_id": [0, 0, 1, 1]}
    The first two chars of document if should correspond 
    to a two-letter genre identifier.
    """
    sentence_ids = []
    words = []
    for i, s in enumerate(sent_lst):
        s_words = re.findall(r"[\w']+|[.,!?;]", s.title())
        words.extend(s_words)
        sentence_ids.extend((np.zeros(len(s_words))+i).astype(int))
        
    data = {
        "document_id": genre_identifier + "_" + file_id,
        "cased_words": words,
        "sent_id":     sentence_ids,
    }
    return data


def text_to_sentence_w_stanza(text):
    """
    incompatible with wl-coref env
    """
    nlp = stanza.Pipeline(lang='en', processors='tokenize')  
    stanza_doc = nlp(text)
    sentences = []
    for sentence in stanza_doc.sentences:
        sentences.append(sentence.text)
    return sentences

def text_to_sentence_w_split(text):
    return [s + "." for s in text.split(".") if len(s)]


def main(): 
    # Paths
    root = '/path/to/ReferenceSpeech/'
    dataset = "dementiabank" #'voxceleb_annotated_usa_concatenated' #'clac_healthy' #'timit_concatenated' 
    metadata_path = root + 'data_info/' + dataset + '.csv'
    features_out_dir = root + '/features/' + dataset

    # mk dir
    os.makedirs(features_out_dir, exist_ok=True)

    # get file paths:
    print ('Listing text files...')
    metadf = pd.read_csv(metadata_path)
    fpaths = metadf.transcript_path.values
    file_ids = metadf.wav_path.values

    # load transcriptions
    print ('Loading transcripts...')
    sentences_lst = []
    for text_path in tqdm(fpaths):
        content = open(text_path).read()
        sentences_lst.append(text_to_sentence_w_split(content))

    # initialize model
    print ('Loading coref analyser...')
    coref_analyser = CorefAnalysis()

    # compute features
    print ('Computing features...')
    res = []
    #for fpath, sentences in tqdm(zip(fpaths, sentences_lst)):
    for fpath, sentences in zip(fpaths, sentences_lst):
        print("processing", fpath, end="\r")
        if len (sentences) and "id07376" not in fpath:
            try:
                feats = {}
                ## count ambiguous pronouns
                d = convert_jsonlines(sentences, genre_identifier="tc")
                (ambiguous_ref_count, ambiguous_ref_ratio, ref_chain_count, 
                ref_chain_ratio) = coref_analyser.compute_ambiguous_refs(d)
                feats["ambiguous_ref_count"] = ambiguous_ref_count
                feats["ambiguous_ref_ratio"] = ambiguous_ref_ratio
                feats["ref_chain_count"] = ref_chain_count
                feats["ref_chain_ratio"] = ref_chain_ratio
                res.append(feats)
            except:
                feat_names = [
                    'ambiguous_ref_count', 'ambiguous_ref_ratio', 
                    'ref_chain_count', 'ref_chain_ratio', 
                    ]
                res.append(make_empty_dict(feat_names))
                print()
                print ("Extraction of coref features was not possible for ", fpath, "with transcript:")
                print (sentences)

        else:
            feat_names = [
                'ambiguous_ref_count', 'ambiguous_ref_ratio', 
                'ref_chain_count', 'ref_chain_ratio', 
                ]
            print()
            res.append(make_empty_dict(feat_names))
            print (fpath, "contains an empty transcript, or consists of arabic speaker id07376")
    
    df = pd.DataFrame(res)
    df['transcript_path'] = fpaths
    df['file'] = file_ids

    outpath = os.path.join(features_out_dir, 'coref.csv')
    print ("Saving features at", outpath)
    df.to_csv(outpath, index=False)
    

if __name__ == '__main__':
    main()
