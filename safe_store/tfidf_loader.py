from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFLoader:
    @staticmethod
    def create_vectorizer_from_dict(tfidf_info):
        vectorizer = TfidfVectorizer(**tfidf_info['params'])
        vectorizer.vocabulary_ = tfidf_info['vocabulary']
        vectorizer.idf_ = [tfidf_info['idf_values'][feature] for feature in vectorizer.get_feature_names()]
        return vectorizer

    @staticmethod
    def create_dict_from_vectorizer(vectorizer):
        tfidf_info = {
            "vocabulary": vectorizer.vocabulary_,
            "idf_values": vectorizer.idf_,# dict(zip(vectorizer.get_feature_names(), )),
            "params": vectorizer.get_params()
        }
        return tfidf_info