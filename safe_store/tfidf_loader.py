import numpy
from safe_store.tf_idf_vectorizer import TfidfVectorizer

class TFIDFLoader:
    @staticmethod
    def create_vectorizer_from_dict(tfidf_info):
        vectorizer = TfidfVectorizer(**tfidf_info['params'])
        vectorizer.ngram_range = tuple(vectorizer.ngram_range)
        vectorizer.vocab = tfidf_info['vocabulary']
        vectorizer.idf = tfidf_info['idf_values']
        dt = vectorizer.dtype[8:-2]
        vectorizer.dtype = eval(dt)
        return vectorizer

    @staticmethod
    def create_dict_from_vectorizer(vectorizer):
        tfidf_info = {
            "vocabulary": vectorizer.vocab,
            "idf_values": vectorizer.idf,# dict(zip(vectorizer.get_feature_names(), )),
            "params": vectorizer.get_params()
        }
        tfidf_info["params"]["dtype"]=str(tfidf_info["params"]["dtype"])

        return tfidf_info
