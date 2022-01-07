import time
import sys
import tomotopy as tp

from utils.pipeline import Pipeline


def calculate_coherence(text_data, max_topic_number, top_n_words):

    # calculate coherence using preset
    with open(coherence, 'w', encoding='utf-8') as fw:
        fw.write('type\ttopic_number\tavg coherence\tcoherence_dist\n')

        for topic_number in range(max_topic_number):
            mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=1, rm_top=5, k=topic_number + 1)
            for doc in text_data:
                mdl.add_doc(doc)

            mdl.burn_in = 100
            mdl.train(0)
            print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs, ', Num words:', mdl.num_words)
            print('Removed top words:', mdl.removed_top_words)
            print('Training...', file=sys.stderr, flush=True)
            for i in range(0, 1500, 10):
                mdl.train(10)
                print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

            for preset in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
                coh = tp.coherence.Coherence(mdl, coherence=preset, top_n=top_n_words)
                avg_coherence = coh.get_score()
                coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]

                print('==== Coherence : {} ===='.format(preset))
                print('Average:', avg_coherence, '\nPer Topic:', coherence_per_topic)

                coherence_per_topic = [str(e)[:9] for e in coherence_per_topic]
                coherence_per_topic = ", ".join(coherence_per_topic)

                fw.write(f'{preset}\t{topic_number + 1}\t{str(avg_coherence)[:9]}\t{coherence_per_topic}\n')

if __name__ == '__main__':

    start = time.time()  # 시작 시간 저장

    max_num_of_topic = 29
    top_n_words = 20

    input_file = './data/abstract.txt'

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
                        top_n_words=top_n_words)

    second = time.time() - start
    print("{}시간 ({}분)".format((second/3600), (second/60)))