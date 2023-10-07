"""
BM25 (Best Matching 25) Explanation:

BM25 is a ranking function used for information retrieval and text search. It is an extension of the TF-IDF (Term Frequency-Inverse Document Frequency) weighting scheme, designed to overcome some of its limitations. BM25 takes into account the distribution of terms in the document corpus and the frequency of terms in a given document.

The key components of BM25 are as follows:

1. TF (Term Frequency): BM25 considers the frequency of each term in a document. It rewards documents that contain more occurrences of query terms.

2. IDF (Inverse Document Frequency): BM25 calculates the inverse document frequency of each term in the corpus. It penalizes terms that appear in a large number of documents and gives more weight to rare terms.

3. Document Length Normalization: BM25 normalizes the term frequency by taking into account the length of the document. Longer documents are penalized less than shorter ones.

4. Query Parameters (k1 and b): BM25 has two tuning parameters, k1 and b, that control the impact of term frequency and document length, respectively. Adjusting these parameters can fine-tune the ranking results.

The BM25 scoring formula is as follows:

BM25(d, Q) = Σ(wi * IDF(ti) * ((fi * (k1 + 1)) / (fi + k1 * (1 - b + (b * |d| / avgdl))))

Where:
- d: Document being scored.
- Q: Query terms.
- ti: Each term in the query Q.
- fi: Term frequency of term ti in document d.
- |d|: Length of document d.
- avgdl: Average document length in the corpus.
- k1 and b: Tuning parameters.

In practice, to find the most relevant documents for a query, you calculate BM25 scores for each document in the corpus with respect to the query, and then rank the documents by their BM25 scores. The higher the BM25 score, the more relevant the document is to the query.

This class, BM25Vectorizer, provides a way to compute BM25 scores for a corpus of text documents and retrieve the most relevant documents for a given query.
"""
import numpy as np
import re

def split_string(doc:str):
    # Split the string using regular expressions to keep punctuation as separate tokens
    words = re.findall(r'\b\w+\b|[.,!?;]', doc)
    return words
class BM25Vectorizer:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_term_freqs = []
        self.doc_count = 0
        self.term_idf = {}

    def fit(self, corpus):
        self.doc_count = len(corpus)
        self.doc_lengths = [len(split_string(doc)) for doc in corpus]
        self.avg_doc_length = np.mean(self.doc_lengths)

        self.doc_term_freqs = []
        for doc in corpus:
            term_freqs = {}
            for term in split_string(doc):
                term_freqs[term] = term_freqs.get(term, 0) + 1
            self.doc_term_freqs.append(term_freqs)

        # Calculate IDF for each term in the corpus
        term_doc_count = {}
        for term_freqs in self.doc_term_freqs:
            for term in term_freqs:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        for term, doc_count in term_doc_count.items():
            self.term_idf[term] = np.log((self.doc_count - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

    def transform(self, query):
        scores = []

        for doc_term_freqs, doc_length in zip(self.doc_term_freqs, self.doc_lengths):
            doc_score = 0
            for term in split_string(query):
                if term in doc_term_freqs:
                    idf_term = self.term_idf[term]
                    tf_term = doc_term_freqs[term]
                    doc_score += (idf_term * (tf_term * (self.k1 + 1))) / (tf_term + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length)))

            scores.append(doc_score)

        return np.array(scores)
