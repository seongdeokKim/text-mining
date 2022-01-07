import sys
import time
import tomotopy as tp

from utils.pipeline import Pipeline


def execute_lda(text_data, topic_number, top_n_words, save_path):

    mdl = tp.LDAModel(tw=tp.TermWeight.ONE,
                      min_cf=1,
                      rm_top=5,
                      k=topic_number)

    docs, corpus = text_data
    for doc in docs:
        mdl.add_doc(doc)


    # LDA model training
    mdl.burn_in = 100
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs, ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for i in range(0, 1500, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

    print('Saving...', file=sys.stderr, flush=True)
    mdl.save(save_path, True)


    # extract topic distribution of each document
    with open(topic_distribution_per_doc, 'w', encoding='utf-8') as fw:
        fw.write('corpus\ttop_words\ttopic_dist\n')

        for i, doc in enumerate(mdl.docs):
            topic_dist = doc.get_topics(top_n=3)
            topic_dist = [str(e[0]) + ', ' + str(e[1])[:8] for e in topic_dist]
            topic_dist = '; '.join(topic_dist)

            top_words_with_prob = doc.get_words()
            top_words = [e[0] for e in top_words_with_prob]
            top_words = ', '.join(top_words)

            fw.write(f'{corpus[i]}\t{top_words}\t{topic_dist}\n')


    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(mdl)


    # ranking the candidates of labels for a specific topic
    with open(words_per_topic, 'w', encoding='utf-8') as fw:
        labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)

        for k in range(mdl.k):
            labels = ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=20))
            fw.write(f'### {k}\t{labels}\n')

            for word, prob in mdl.get_topic_words(k, top_n=top_n_words):
                fw.write(f'{word}\t{prob}\n')

def infer_unseen_document(model_file, unseen_words):
    from nltk import word_tokenize

    mdl = tp.LDAModel.load(model_file)

    doc_inst = mdl.make_doc(word_tokenize(unseen_words))
    topic_dist, ll = mdl.infer(doc_inst)
    print("Topic Distribution for Unseen Doc: ", topic_dist)
    print("Log-likelihood of inference: ", ll)
    print()


if __name__ == '__main__':

    start = time.time()  # 시작 시간 저장

    topic_number = 29
    save_path = './lda/lda.bin'
    top_n_words = 20

    # input_file = './data/reference_abstract_13630.txt'
    input_file = './data/abstract.txt'

    coherence = './lda/coherence_max_k_' + str(topic_number) + '.txt'
    words_per_topic = './lda/words_per_topic_k_' + str(topic_number) + '.txt'
    topic_distribution_per_doc = './lda/topic_distribution_per_doc_k_' + str(topic_number) + '.txt'


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

    print('Running LDA')
    execute_lda(text_data=(docs, corpus),
                topic_number=topic_number,
                top_n_words=top_n_words,
                save_path=save_path)


    # test
    unseen_text='아사이 베리 블루베리 비슷하다'
    infer_unseen_document(model_file=save_path,
                          unseen_words=unseen_text)


    second = time.time() - start
    print("{}시간 ({}분)".format((second/3600), (second/60)))