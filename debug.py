import os
import pickle
import re
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import logging
import keras

from dataset import get_matrices, get_duc, shuffle_data, get_nyt, \
    store_pas_nyt_dataset, compute_idfs, store_matrices, get_nyt_pas_lists, arrange_nyt_pas_lists, get_duc_pas_lists
from loss_testing import summary_clustering_score, summary_clustering_score_2
from pas import realize_pas
from summarization import testing, testing_weighted, rouge_score, build_model, train_model, best_pas, generate_summary
from utils import sentence_embeddings, plot_history, get_sources_from_pas_lists, sample_summaries, result_path

_duc_path_ = os.getcwd() + "/dataset/duc_source"
_nyt_path_ = "D:/Datasets/nyt_corpus/data"

for i in range(35):
    print("refactoring batch: " + str(i))
    docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=0)
    print("docs...")
    for pas_list in docs_pas_lists:
        for pas in pas_list:
            pas.realized_pas = realize_pas(pas)
    print("refs...")
    for pas_list in refs_pas_lists:
        for pas in pas_list:
            pas.realized_pas = realize_pas(pas)

    with open(os.getcwd() + "/dataset/nyt/compact/compact_nyt_docs_pas" + str(i) + ".dat", "wb") as dest_f:
        pickle.dump(docs_pas_lists, dest_f)
    with open(os.getcwd() + "/dataset/nyt/compact/compact_nyt_refs_pas" + str(i) + ".dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)

"""  SUMMARIES CHECK
doc_matrix, ref_matrix, score_matrix = get_matrices((0.0, 1.0), index=0)
docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=0)
refs = get_sources_from_pas_lists(refs_pas_lists)

recall_scores_list = [0] * 1000
summaries = []

max_sent_no = doc_matrix.shape[1]

for pas_list in docs_pas_lists:
    for pas in pas_list:
        pas.realized_pas = realize_pas(pas)

summaries = []
for i in range(len(docs_pas_lists)):
    pas_list = docs_pas_lists[i]
    pas_no = len(pas_list)
    sent_vec_len = len(pas_list[0].vector) + len(pas_list[0].embeddings)

    pred_scores = score_matrix[i, :]
    scores = pred_scores[:pas_no]
    summary = generate_summary(pas_list, scores)
    summaries.append(summary)
sample_summaries("maximum_scores", docs_pas_lists, refs, recall_scores_list, summaries=summaries, batch=0)
"""

"""     TESTING WEIGHTED PAS METHOD (SIMPLE)
weights_list = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.2, 0.1, 0.3, 0.1, 0.1, 0.2], [0.3, 0.05, 0.3, 0.1, 0.05, 0.2], [0.1, 0.05, 0.3, 0.1, 0.15, 0.2]]

batches = 6
training_no = 666

rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                "rouge_2_precision": 0, "rouge_2_f_score": 0}


for weights in weights_list:
    for index in range(batches):
        docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=index)
        refs = get_refs_from_pas(refs_pas_lists[training_no:])

        score = testing_weighted(docs_pas_lists[training_no:], refs, weights)
        rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
        rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
        rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
        rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
        rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
        rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

    for k in rouge_scores.keys():
        rouge_scores[k] /= batches

    with open(_result_path_, "a") as res_file:
        print(weights, file=res_file)
        print(rouge_scores, file=res_file)
        print("=================================================", file=res_file)
"""

"""        COMPUTING MAXIMUM SCORES (PER SCORING METHOD)
weights_list = [(0.0, 1.0),
                #(0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                #(0.4, 0.6), (0.5, 0.5),(0.6, 0.4), (0.7, 0.3),
                #(0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
                ]
for weights in weights_list:
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}
    batches = 1

    for k in range(batches):
        doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, index=k)
        docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=k)
        refs = get_sources_from_pas_lists(refs_pas_lists)
        # _, refs, _ = get_duc(_duc_path_)       DUC

        training_no = 666       # includes validation.

        docs_pas_lists = docs_pas_lists[training_no:]
        doc_matrix = doc_matrix[training_no:, :, :]
        score_matrix = score_matrix[training_no:, :]
        refs = refs[training_no:]

        recall_scores_list = []
        summaries = []

        max_sent_no = doc_matrix.shape[1]

        for pas_list in docs_pas_lists:
            for pas in pas_list:
                pas.realized_pas = realize_pas(pas)

        for i in range(len(docs_pas_lists)):
           # print(weights)
           # print(k)
           # print("Processing doc:" + str(i) + "/" + str(len(docs_pas_lists)))
            pas_list = docs_pas_lists[i]
            pas_no = len(pas_list)
            sent_vec_len = len(pas_list[0].vector) + len(pas_list[0].embeddings)

            pred_scores = score_matrix[i, :]
            scores = pred_scores[:pas_no]
            summary = generate_summary(pas_list, scores)

            score = rouge_score([summary], [refs[i]])

            summaries.append(summary)
            recall_scores_list.append(score["rouge_1_recall"])

            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

        sample_summaries("maximum_scores", docs_pas_lists, refs, recall_scores_list, summaries=summaries, batch=k)

    for k in rouge_scores.keys():
        rouge_scores[k] /= 334 * batches

    with open(result_path + "results.txt", "a") as res_file:
        print("maximum score" + str(weights), file=res_file)
        print(rouge_scores, file=res_file)
        print("=================================================", file=res_file)


"""

"""        TESTING & TRAINING NYT
tst = True
trn = False
binary = False
weights_list = [  # (0.0, 1.0),
    # (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
    # (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
    # (0.8, 0.2), (0.9, 0.1),
    (1.0, 0.0)]

batches = 15
# training_no = 666                   # includes validation.
training_no = 0
doc_size = 300
vector_size = 134

for weights in weights_list:
    model_name = "nyt_001_10_4_" + str(weights)
    save_model = False

    if trn:
        model = build_model(doc_size, vector_size)
        for index in range(batches):
            doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, binary=binary, index=index)
            docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index)
            refs = get_refs_from_pas(refs_pas_lists)

            if index == batches - 1:
                save_model = True

            print(weights)
            print("index: " + str(index))
            train_model(model, model_name, doc_matrix[:training_no, :, :], score_matrix[:training_no, :], epochs=4,
                        batch_size=50, save_model=save_model)

    if tst:
        # rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
        #                "rouge_2_precision": 0, "rouge_2_f_score": 0}
        # for index in range(batches):
        doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, binary=binary)
        # doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, binary=binary, index=index)
        docs_pas_lists, refs_pas_lists = get_duc_pas_lists()
        # docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index)
        _, refs, _ = get_duc(_duc_path_)
        # refs = get_refs_from_pas(refs_pas_lists)

        # score = testing(model_name,
        rouge_scores = testing(model_name,
                               docs_pas_lists[training_no:],
                               doc_matrix[training_no:, :doc_size, :],
                               refs[training_no:])

        # rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
        # rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
        # rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
        # rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
        # rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
        # rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

        # for k in rouge_scores.keys():
        #   rouge_scores[k] /= batches

        with open(_result_path_, "a") as res_file:
            print(model_name + "DUC_TEST", file=res_file)
            print(rouge_scores, file=res_file)
            print("=================================================", file=res_file)

"""


"""# CLUSTERING TEST, ONE BY ONE

# Position score, sentence length score, tf_idf, numerical data, centrality, title.
#weights = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
#weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
weights = [0.2, 0.1, 0.3, 0.1, 0.1, 0.2]                # best so far.
#weights = [0.3, 0.05, 0.3, 0.1, 0.05, 0.2]
index = 0
docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=index)
doc_matrix, ref_matrix, _ = get_matrices((0.1, 0.9), index=index)
doc_range = len(docs_pas_lists)

outliers = []
doc_perc_tot = 0
ref_perc_tot = 0
best_perc_tot = 0
adj_score_tot = 0
score_tot = 0
for i in range(doc_range):
    print(i)
    best_pas_list = best_pas(docs_pas_lists[i], len(refs_pas_lists[i]), weights)
    #best_vectors = np.array([np.append(pas.vector, pas.embeddings) for pas in best_pas_list])
    best_vectors = np.array([np.array(pas.vector) for pas in best_pas_list])

    doc_perc, ref_perc, best_perc, score, adj_score = summary_clustering_score(doc_matrix[i, :, :6], best_vectors, ref_matrix[i, :, :6], log=False)
    if doc_perc:
        doc_perc_tot += doc_perc
        ref_perc_tot += ref_perc
        best_perc_tot += best_perc
        adj_score_tot += adj_score
        score_tot += score
    else:
        outliers.append(i)

doc_no = len(docs_pas_lists) - len(outliers)
print("Outliers: " + str(outliers))
print(weights)
print("FINAL SCORES:")
print("DOC %:")
print("{:.3%}".format(doc_perc_tot/doc_no))
print("REF %:")
print("{:.3%}".format(ref_perc_tot/doc_no))
print("BEST %:")
print("{:.3%}".format(best_perc_tot/doc_no))
print("SCORE:")
print("{:.3%}".format(score_tot/doc_no))
print("ADJUSTED SCORE:")
print("{:.3%}".format(adj_score_tot/doc_no))

"""

""" CLUSTERING TEST MULTIPLE CLUSTERS
# Position score, sentence length score, tf_idf, numerical data, centrality, title.
#weights = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
#weights = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
#weights = [0.2, 0.1, 0.3, 0.1, 0.1, 0.2]
weights = [0.4, 0.1, 0.1, 0.3, 0.0, 0.1]        # best so far.
index = 1
docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=index)
doc_matrix, ref_matrix, _ = get_matrices((0.1, 0.9), index=index)


index = 0
summ_coverage_tot = 0
summ_adj_coverage_tot = 0

for index in range(len(docs_pas_lists)):
    best_pas_list = best_pas(docs_pas_lists[index], len(refs_pas_lists[index]), weights)
    best_vectors = np.array([np.array(pas.embeddings) for pas in best_pas_list])

    summ_coverage, summ_adj_coverage = summary_clustering_score_2(doc_matrix[index, :, 6:], best_vectors, ref_matrix[index, :, 6:])
    summ_coverage_tot += summ_coverage
    summ_adj_coverage_tot += summ_adj_coverage

print("\n\n\n\n")
print("AVERAGE SUMMARY COVERAGE:")
print(summ_coverage_tot/len(docs_pas_lists))

print("AVERAGE SUMMARY ADJUSTED COVERAGE:")
print(summ_adj_coverage_tot/len(docs_pas_lists))

"""

""" # CLUSTERING USING ALL THE DOCS
# Position score, sentence length score, tf_idf, numerical data, centrality, title.
#weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
#weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
#weights = [0.2, 0.1, 0.3, 0.1, 0.1, 0.2]                # best so far.
#weights = [0.3, 0.05, 0.3, 0.1, 0.05, 0.2]

docs_pas_lists, refs_pas_lists = get_pas_lists()
doc_matrix, ref_matrix, _  = get_matrices(include_embeddings=False)

all_doc_vectors = doc_matrix[0, :, :]
all_ref_vectors = ref_matrix[0, :, :]

best_pas_list = best_pas(docs_pas_lists[0], len(refs_pas_lists[0]), weights)
# best_vectors = np.array([np.append(pas.vector, pas.embeddings) for pas in best_pas_list])
all_best_vectors = np.array([np.array(pas.vector) for pas in best_pas_list])

for i in range(1, len(docs_pas_lists)):
    all_doc_vectors = np.concatenate((all_doc_vectors, doc_matrix[i]))
    all_ref_vectors = np.concatenate((all_ref_vectors, ref_matrix[i]))


    best_pas_list = best_pas(docs_pas_lists[i], len(refs_pas_lists[i]), weights)
    #best_vectors = np.array([np.append(pas.embeddings, pas.vector) for pas in best_pas_list])
    best_vectors = np.array([np.array(pas.vector) for pas in best_pas_list])
    all_best_vectors = np.concatenate((all_best_vectors, best_vectors))


doc_perc, ref_perc, best_perc, score = summary_clustering_score(all_doc_vectors, all_best_vectors, all_ref_vectors)

print(weights)
print("FINAL SCORES:")
print("DOC %:")
print("{:.3%}".format(doc_perc))
print("REF %:")
print("{:.3%}".format(ref_perc))
print("BEST %:")
print("{:.3%}".format(best_perc))
print("SCORE:")
print("{:.3%}".format(score))

"""
