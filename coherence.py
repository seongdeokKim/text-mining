import time
import sys
import tomotopy as tp

from utils.pipeline import Pipeline
from lda import LDAModal


class ModelConfig:
    def __init__(self, min_cf,
                 remove_top_n_words,
                 top_n_words,
                 top_n_topic_labels):

        self._min_cf = min_cf
        self._remove_top_n_words = remove_top_n_words
        self._top_n_words = top_n_words
        self._top_n_topic_labels = top_n_topic_labels
        self._topic_number = None

    @property
    def min_cf(self):
        return self._min_cf

    @property
    def remove_top_n_words(self):
        return self._remove_top_n_words

    @property
    def top_n_words(self):
        return self._top_n_words

    @property
    def top_n_topic_labels(self):
        return self._top_n_topic_labels

    @property
    def topic_number(self):
        return self._topic_number

    @topic_number.setter
    def topic_number(self, value: int):
        if value < 0 or value > 100:
            raise ValueError("Check current_topic_number argument! something might be wrong.")
        self._topic_number = value


def calculate_coherence(text_data,
                        max_topic_number,
                        coherence_path,
                        model_config):
    """calculate coherence based on four scheme type"""

    # iterate calculation as the number of topic increases
    coherence_list = []
    for current_topic_number in range(1, max_topic_number):
        model_config.topic_number = current_topic_number

        # declare LDAModel class
        lda_model = LDAModal(min_cf=model_config.min_cf,
                             remove_top_n_words=model_config.remove_top_n_words,
                             top_n_words=model_config.top_n_words,
                             top_n_topic_labels=model_config.top_n_topic_labels,
                             topic_number=model_config.topic_number)

        # put documents to lda_model instance
        for doc in text_data:
            lda_model.mdl.add_doc(doc)

        # model train
        lda_model.train()

        # calculate coherence of each scheme
        coherences = lda_model.get_coherence()
        coherence_list += coherences

    # file writer
    with open(coherence_path, 'w', encoding='utf-8') as fw:
        fw.write('{}\t{}\t{}\t{}\n'.format("scheme_type", "topic_number", "average coherence", "coherence_dist"))
        for c in coherence_list:
            fw.write(f'{c.scheme}\t{c.topic_number}\t{c.average_coherence}\t{c.coherence_of_each_topic}\n')


if __name__ == '__main__':

    start = time.time()  # 시작 시간 저장

    config = ModelConfig(min_cf=1,
                         remove_top_n_words=5,
                         top_n_words=20,
                         top_n_topic_labels=5)


    input_file = './data/abstract.txt'
    max_num_of_topic = 30
    coherence = './coherence_max_' + str(max_num_of_topic) + '.txt'

    # load documents on memory
    corpus = []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for doc in fr:
            corpus.append(doc.strip())

    # preprocess corpus
    pipeline = Pipeline()
    result = pipeline.preprocess_corpus(corpus)

    # transforming pipeline output format into DMR input format
    docs = []
    for doc in result:
        new_doc = []
        for sent in doc:
            for token in sent:
                new_doc.append(token.lower().strip())
        docs.append(new_doc)


    print('Calculating Coherence')

    calculate_coherence(text_data=docs,
                        max_topic_number=max_num_of_topic,
                        coherence_path=coherence,
                        model_config=config)

    second = time.time() - start
    print("{}시간 ({}분)".format((second/3600), (second/60)))