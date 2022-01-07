from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import nltk

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

class Pipeline:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = self.get_stopwords(file='./stopwords/stopwordsEng.txt')

    def preprocess_corpus(self, corpus: 'list of str') -> 'list of list of str':

        preprocessed_corpus = []
        for doc in corpus:

            preprocessed_doc = []
            for sent in sent_tokenize(doc):
                preprocessed_doc.append(self.preprocess_sent(sent))

            preprocessed_corpus.append(preprocessed_doc)

        return preprocessed_corpus

    def preprocess_corpus_with_metadata(self, corpus: 'list of str', metadata: 'list of str') \
            -> 'tuple(list of list of str, list of str)':

        preprocessed_corpus = []
        pair_map = []
        for doc, meta in zip(corpus, metadata):

            preprocessed_doc = []
            for sent in sent_tokenize(doc):
                preprocessed_doc.append(self.preprocess_sent(sent))

            preprocessed_corpus.append(preprocessed_doc)
            pair_map.append(meta.strip())

        return (preprocessed_corpus, pair_map)


    def preprocess_sent(self, sent: 'str') -> 'list of str':

        # word tokenize
        tokenized = word_tokenize(sent)

        # POS tagging
        tagged = pos_tag(tokenized)

        # POS filtering
        tagged = [token_and_tag for token_and_tag in tagged if "NN" in token_and_tag[1]]

        # lemmatize word
        lemmatized = self.lemmatize(tagged)

        # select only word
        words = [token_and_tag[0] for token_and_tag in lemmatized]

        # filter stopword
        result = [word for word in words if word not in self.stopwords]

        return result

    def lemmatize(self, tagged_sent: 'list of tuple') -> 'list of tuple':

        _tagged_sent = []
        for token_and_tag in tagged_sent:
            word = self.lemmatizer.lemmatize(token_and_tag[0], 'n')
            _tagged_sent.append((word, token_and_tag[1]))

        return _tagged_sent

    def get_stopwords(self, file='./stopwords/stopwordsEng.txt'):

        with open(file, 'r', encoding='utf-8') as fr:
            stopwords = [stopword.strip() for stopword in fr]

        return set(stopwords)