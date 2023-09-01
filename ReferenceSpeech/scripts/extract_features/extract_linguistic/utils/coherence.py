import numpy as np
from sentence_transformers import SentenceTransformer, util

# Adapted from https://www.sbert.net/docs/quickstart.html

class SemanticSimilarityScorer:    
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_features(self, sentence_lst):
        embeddings = self.model.encode(sentence_lst)
        adjacent_cosine_sim = self.adjacent_cosine_similarity(embeddings)
        mean_similarity = self.mean_similarity(adjacent_cosine_sim)
        ongoing_variability = self.ongoing_variability(adjacent_cosine_sim)
        return embeddings, mean_similarity, ongoing_variability

    def encode_embeddings(self, sentence_lst):
        sentence_embeddings = self.model.encode(sentence_lst)
        return sentence_embeddings    

    def adjacent_cosine_similarity(self, embeddings):
        #Compute cosine similarity between all pairs
        cos_sim = util.cos_sim(embeddings, embeddings)

        adjacent_similarities = []
        for i in range(len(cos_sim)-2):
            adjacent_similarities.append(cos_sim[i][i+1])
        
        return adjacent_similarities
    
    def ongoing_variability(self, adjacent_sim):
        """
        measures the variability. If all sentences are equally dissimilar to 
        each other, the variablity willt be very low. Motivated by 
        [Sanz et al., 2022, Automated text-level semantic markers of 
        Alzheimer's disease].
        """
        a = np.square(adjacent_sim - np.mean(adjacent_sim))
        ongoing_variability = np.sum(a)/len(a) 
        return ongoing_variability

    def mean_similarity(self, adjacent_sim):
        return np.mean(adjacent_sim)


def text_segmenter_n_tokens(text, n_tokens=14):
    words = text.split()
    segments = [
        " ".join(words[i:i+n_tokens]) for i in range(0, len(words), n_tokens)]
    return segments