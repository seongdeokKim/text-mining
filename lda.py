import sys
import time
import tomotopy as tp
from nltk import word_tokenize

from utils.pipeline import Pipeline


class LDAModal:
    def __init__(self, min_cf,
                 remove_top_n_words,
                 topic_number,
                 top_n_words,
                 top_n_topic_labels):

        self.min_cf = min_cf
        self.remove_top_n_words = remove_top_n_words
        self.topic_number = topic_number
        self.top_n_words = top_n_words
        self.top_n_topic_labels = top_n_topic_labels

        self.mdl = tp.LDAModel(tw=tp.TermWeight.ONE,
                               min_cf=self.min_cf,
                               rm_top=self.top_n_topic_labels,
                               k=self.topic_number)

    def execute_lda(self, text_data,
                    topic_dist_per_doc_path,
                    words_per_topic_path,
                    save_path):

        docs, corpus = text_data
        for doc in docs:
            self.mdl.add_doc(doc)

        self.train(save_path)
        self.extract_topic_dist_per_doc(topic_dist_per_doc_path)
        self.rank_top_words_per_topic(words_per_topic_path)


    def train(self, save_path=None):
        """LDA model training"""

        self.mdl.burn_in = 100
        self.mdl.train(0)
        print('Num docs:', len(self.mdl.docs))
        print('Vocab size:', self.mdl.num_vocabs, ', Num words:', self.mdl.num_words)
        print('Removed top words:', self.mdl.removed_top_words)
        print('Training...', file=sys.stderr, flush=True)
        for i in range(0, 1500, 10):
            self.mdl.train(10)
            print('Iteration: {}\tLog-likelihood: {}'.format(i, self.mdl.ll_per_word))

        if save_path is not None:
            print('Saving...', file=sys.stderr, flush=True)
            self.mdl.save(model_path, True)

    def extract_topic_dist_per_doc(self, topic_dist_per_doc_path):
        """extract topic distribution of each document"""

        with open(topic_dist_per_doc_path, 'w', encoding='utf-8') as fw:
            fw.write('corpus\ttop_words\ttopic_dist\n')

            for i, doc in enumerate(self.mdl.docs):
                topic_dist = doc.get_topics(top_n=3)
                topic_dist = [str(e[0]) + ', ' + str(e[1])[:8] for e in topic_dist]
                topic_dist = '; '.join(topic_dist)

                top_words_with_prob = doc.get_words()
                top_words = [e[0] for e in top_words_with_prob]
                top_words = ', '.join(top_words)

                fw.write(f'{corpus[i]}\t{top_words}\t{topic_dist}\n')

    def rank_top_words_per_topic(self, words_per_topic_path):
        """extract probable words per topic"""

        # extract candidates for auto topic labeling
        extractor = tp.label.PMIExtractor(min_cf=10,
                                          min_df=5,
                                          max_len=5,
                                          max_cand=10000)

        cands = extractor.extract(self.mdl)

        # ranking the candidates of labels for a specific topic
        with open(words_per_topic_path, 'w', encoding='utf-8') as fw:
            labeler = tp.label.FoRelevance(self.mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)

            for k in range(self.mdl.k):
                labels = ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=self.top_n_topic_labels))
                fw.write(f'### {k}\t{labels}\n')

                for word, prob in self.mdl.get_topic_words(k, top_n=self.top_n_words):
                    fw.write(f'{word}\t{prob}\n')

    def infer_unseen_document(self, saved_model, unseen_words):
        '''infer unseen document through trained LDA model'''

        mdl = tp.LDAModel.load(saved_model)

        doc_inst = mdl.make_doc(word_tokenize(unseen_words))
        topic_dist, ll = mdl.infer(doc_inst)
        print("Topic Distribution for Unseen Doc: ", topic_dist)
        print("Log-likelihood of inference: ", ll)
        print()

    def get_coherence(self):
        # calculate coherence of each scheme

        coherence_list = []
        for type in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
            coh = tp.coherence.Coherence(self.mdl, coherence=type, top_n=self.top_n_words)

            avg_coherence = coh.get_score()
            avg_coherence = str(avg_coherence)[:9]

            coherence_per_topic = [coh.get_score(topic_id=k) for k in range(self.mdl.k)]
            coherence_per_topic = [str(e)[:9] for e in coherence_per_topic]
            coherence_per_topic = ", ".join(coherence_per_topic)

            print('==== Coherence : {} ===='.format(type))
            print('Average:', avg_coherence, '\nPer Topic:', coherence_per_topic)

            coherence = Coherence(scheme=type,
                                  topic_number=self.topic_number,
                                  average_coherence=avg_coherence,
                                  coherence_of_each_topic=coherence_per_topic)

            coherence_list.append(coherence)

        return coherence_list

class Coherence:
    def __init__(self, scheme,
                 topic_number,
                 average_coherence,
                 coherence_of_each_topic):

        self._scheme = scheme
        self._topic_number = topic_number
        self._average_coherence = average_coherence
        self._coherence_of_each_topic = coherence_of_each_topic

    @property
    def scheme(self):
        return self._scheme

    @property
    def topic_number(self):
        return self._topic_number

    @property
    def average_coherence(self):
        return self._average_coherence

    @property
    def coherence_of_each_topic(self):
        return self._coherence_of_each_topic


if __name__ == '__main__':

    start = time.time()  # 시작 시간 저장

    number_of_topic = 10
    model_path = './lda/lda.bin'
    top_k_words = 20
    min_collection_frequency = 1
    remove_top_k_words = 5
    top_k_topic_labels = 5

    input_file = './data/abstract.txt'

    words_per_topic = './lda/words_per_topic_k_' + str(number_of_topic) + '.txt'
    topic_dist_per_doc = './lda/topic_distribution_per_doc_k_' + str(number_of_topic) + '.txt'


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


    lda_model = LDAModal(min_cf=min_collection_frequency,
                         remove_top_n_words=remove_top_k_words,
                         topic_number=number_of_topic,
                         top_n_words=top_k_words,
                         top_n_topic_labels=top_k_topic_labels)

    lda_model.execute_lda(text_data=(docs, corpus),
                          topic_dist_per_doc_path=topic_dist_per_doc,
                          words_per_topic_path=words_per_topic,
                          save_path=model_path)

    # test
    unseen_text='아사이 베리 블루베리 비슷하다'
    lda_model.infer_unseen_document(saved_model=model_path,
                                    unseen_words=unseen_text)


    second = time.time() - start
    print("{}시간 ({}분)".format((second/3600), (second/60)))