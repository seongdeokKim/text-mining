import sys
import time
import tomotopy as tp
from nltk import word_tokenize

from utils.pipeline import Pipeline


class LDAModal:
    def __init__(self, min_cf,
                 remove_top_words,
                 topic_number):

        self.mdl = tp.LDAModel(tw=tp.TermWeight.ONE,
                               min_cf=min_cf,
                               rm_top=remove_top_words,
                               k=topic_number)

    def execute_lda(self, text_data,
                    top_n_words,
                    save_path,
                    topic_dist_per_doc_path,
                    words_per_topic_path):

        docs, corpus = text_data
        for doc in docs:
            self.mdl.add_doc(doc)

        self.train(save_path)
        self.extract_topic_dist_per_doc(topic_dist_per_doc_path)
        self.rank_top_words_per_topic(top_n_words, words_per_topic_path)


    def train(self, model_path):
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

    def rank_top_words_per_topic(self, top_n_words, words_per_topic_path):
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
                labels = ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=20))
                fw.write(f'### {k}\t{labels}\n')

                for word, prob in self.mdl.get_topic_words(k, top_n=top_n_words):
                    fw.write(f'{word}\t{prob}\n')

    def infer_unseen_document(self, saved_model, unseen_words):
        '''infer unseen document through trained LDA model'''

        mdl = tp.LDAModel.load(saved_model)

        doc_inst = mdl.make_doc(word_tokenize(unseen_words))
        topic_dist, ll = mdl.infer(doc_inst)
        print("Topic Distribution for Unseen Doc: ", topic_dist)
        print("Log-likelihood of inference: ", ll)
        print()


if __name__ == '__main__':

    start = time.time()  # 시작 시간 저장

    topic_number = 10
    save_path = './lda/lda.bin'
    top_n_words = 20
    min_collection_frequency = 1
    remove_top_words = 5

    input_file = './data/abstract.txt'

    words_per_topic = './lda/words_per_topic_k_' + str(topic_number) + '.txt'
    topic_dist_per_doc = './lda/topic_distribution_per_doc_k_' + str(topic_number) + '.txt'


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
                         remove_top_words=remove_top_words,
                         topic_number=topic_number)

    lda_model.execute_lda(text_data=(docs, corpus),
                          top_n_words=top_n_words,
                          save_path=save_path,
                          topic_dist_per_doc_path=topic_dist_per_doc,
                          words_per_topic_path=words_per_topic)

    # test
    unseen_text='아사이 베리 블루베리 비슷하다'
    lda_model.infer_unseen_document(saved_model=save_path,
                                    unseen_words=unseen_text)


    second = time.time() - start
    print("{}시간 ({}분)".format((second/3600), (second/60)))