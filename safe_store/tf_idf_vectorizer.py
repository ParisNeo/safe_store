import math
from collections import defaultdict

class TfidfVectorizer:
    def __init__(self):
        self.document_count = 0
        self.df = defaultdict(int)
        self.idf = {}
        self.vocab = {}
        self.vocab_inv = {}

    def fit(self, documents):
        self.document_count = len(documents)
        
        for document in documents:
            term_seen = set()
            for term in document.split():
                if term not in term_seen:
                    self.df[term] += 1
                    term_seen.add(term)
        
        for term, freq in self.df.items():
            self.idf[term] = math.log(self.document_count / (1 + freq))
        
        self.vocab = {term: i for i, term in enumerate(self.df.keys())}
        self.vocab_inv = {i: term for term, i in self.vocab.items()}

    def transform(self, documents, default_oov_value=0.1):
        document_vectors = []
        
        for document in documents:
            tf = defaultdict(int)
            for term in document.split():
                if term in self.vocab:
                    tf[term] += 1
                else:
                    # Assign a small default value to OOV words
                    tf[term] = default_oov_value
            
            tf_idf = {self.vocab.get(term, len(self.vocab)): (tf[term] / len(document.split())) * self.idf.get(term, default_oov_value) for term in tf.keys()}
            document_vectors.append(tf_idf)
        
        return document_vectors

    def get_feature_names(self):
        return [self.vocab_inv[i] for i in range(len(self.vocab))]

    def cosine_similarity(self, doc_vector1, doc_vector2):
        """
        Calculates the cosine similarity between two document vectors.
        
        :param doc_vector1: TF-IDF vector for document 1.
        :param doc_vector2: TF-IDF vector for document 2.
        :return: Cosine similarity score.
        """
        # Calculate dot product
        dot_product = sum(doc_vector1.get(i, 0) * doc_vector2.get(i, 0) for i in range(len(self.vocab)))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(val ** 2 for val in doc_vector1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in doc_vector2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            # One of the documents has no terms after processing
            return 0.0
        
        # Calculate cosine similarity
        return dot_product / (magnitude1 * magnitude2)
