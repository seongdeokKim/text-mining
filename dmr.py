import sys
import time
import tomotopy as tp
import pandas as pd
import numpy as np

from utils.pipeline import Pipeline


def execute_dmr(text_data, pair_map, topic_number, top_n_words, save_path):

    mdl = tp.DMRModel(tw=tp.TermWeight.ONE,
                      min_cf=3,
                      rm_top=5,
                      k=topic_number)

    docs, corpus = text_data
    for i, doc in enumerate(docs):
        year = pair_map[i]
        mdl.add_doc(doc, metadata=year)

    # DMR model training
    mdl.burn_in = 100
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs, ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for i in range(0, 1000, 10):
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


    # extracting the candidates of labels and words per each topic
    with open(words_per_topic, 'w', encoding='utf-8') as fw:
        labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)

        for k in range(mdl.k):
            labels = ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=20))
            fw.write(f'### {k}\t{labels}\n')

            for word, prob in mdl.get_topic_words(k, top_n=top_n_words):
                fw.write(f'{word}\t{prob}\n')
            fw.write('\n')


    # calculate metadata distribution per topic
    metadata_list = [mdl.metadata_dict[m] for m in range(mdl.f)]
    metadata_list = sorted(metadata_list, reverse=False)

    df_metadata_dist_per_topic = pd.DataFrame()
    for k in range(mdl.k):  # k: the number of topics

        print('Topic #{}'.format(k))
        metadata_and_value_dict = {}
        for m in range(mdl.f):  # f: the number of metadata
            metadata = mdl.metadata_dict[m]
            value = mdl.lambdas[k][m]

            metadata_and_value_dict[metadata] = value

        sorted_dict = sorted(metadata_and_value_dict.items(), reverse=False)
        values = [e[1] for e in sorted_dict]

        _values = np.array(values)
        median = np.median(_values)
        max = np.max(_values)
        min = np.min(_values)

        for metadata in metadata_list:
            value = metadata_and_value_dict[metadata]
            final_value = abs(max) + value + abs(median)

            metadata_and_value_dict[metadata] = final_value

        sorted_dict = sorted(metadata_and_value_dict.items(), reverse=False)
        new_values = [e[1] for e in sorted_dict]
        print('metadata_list :', metadata_list)
        print('calculated_values :', new_values)
        print()

        df_metadata_dist_per_topic = df_metadata_dist_per_topic.append(pd.Series(new_values), ignore_index=True)

    # convert dataframe to csv file
    df_metadata_dist_per_topic.columns = metadata_list
    df_metadata_dist_per_topic.to_csv(metadata_dist_per_topic, sep=',', encoding='utf-8')


if __name__ == '__main__':

    start = time.time()  # 시작 시간 저장

    topic_number = 10
    top_n_words = 20
    input_file = './data/abstract_with_year.txt'

    words_per_topic = './dmr/words_per_topic_k_' + str(topic_number) + '.txt'
    topic_distribution_per_doc = './dmr/topic_distribution_per_doc_k_' + str(topic_number) + '.txt'
    metadata_dist_per_topic = './dmr/metadata_dist_per_topic_k_' + str(topic_number) + '.csv'

    # load documents on memory
    corpus = []
    metadata_pair = []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            field_line = line.strip().split("\t")

            try: document = field_line[1].strip()
            except: continue

            try: metadata = field_line[0].strip()
            except: continue

            corpus.append(document)
            metadata_pair.append(metadata)

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

    print('Running DMR')
    save_path = './dmr/dmr.bin'
    execute_dmr(text_data=(docs, corpus),
                pair_map=metadata_pair,
                topic_number=topic_number,
                top_n_words=top_n_words,
                save_path=save_path)

    second = time.time() - start
    print("{}시간 ({}분)".format((second/3600), (second/60)))

