import numpy
from safe_store.tf_idf_vectorizer import TfidfVectorizer

class TFIDFLoader:
    @staticmethod
    def create_vectorizer_from_dict(tfidf_info):
        vectorizer = TfidfVectorizer()
        vectorizer.df = tfidf_info['df']
        vectorizer.idf = tfidf_info['idf']
        vectorizer.document_count = tfidf_info['document_count']
        vectorizer.vocab = tfidf_info['vocab']
        vectorizer.vocab_inv = tfidf_info['vocab_inv']
        return vectorizer

    @staticmethod
    def create_dict_from_vectorizer(vectorizer: TfidfVectorizer):
        tfidf_info = {
            "df": vectorizer.df,
            "idf": vectorizer.idf,# dict(zip(vectorizer.get_feature_names(), )),
            "document_count": vectorizer.document_count,
            "vocab": vectorizer.vocab,
            "vocab_inv": vectorizer.vocab_inv
        }

        return tfidf_info
