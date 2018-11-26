import os
import pickle
import re
import time
from math import sqrt

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import logging
import keras
from scipy.stats import stats

from dataset_scores import store_full_sentence_matrices, store_score_matrices, get_matrices
from dataset_text import get_duc, get_nyt, \
    store_pas_nyt_dataset, compute_idfs, get_pas_lists, arrange_nyt_pas_lists
from deep_model import build_model, train_model
from loss_testing import summary_clustering_score, summary_clustering_score_2
from pas import realize_pas
from summarization import best_pas, generate_summary, \
    predict_scores, generate_extract_summary, document_rouge_scores, dataset_rouge_scores_weighted, \
    dataset_rouge_scores_weighted_extractive
from train import train
from utils import sentence_embeddings, get_sources_from_pas_lists, sample_summaries, direct_speech_ratio, timer, tokens, \
    resolve_anaphora, resolve_anaphora_pas_list, direct_speech_ratio_pas

_duc_path_ = os.getcwd() + "/dataset/duc_source"
_nyt_path_ = "D:/Datasets/nyt_corpus/data"

#"""         # ORDER ACCURACY
def dist(vec1, vec2):
    i = 0
    tot_matches_len = 0
    while i < len(vec1):
        match_len = 0
        for j in range(len(vec2)):
            if vec1[i] == vec2[j]:
                match_len += 1
                if i == len(vec1) - 1:
                    break
                i += 1
            elif match_len > 0:
                break
        if match_len > 1:
            tot_matches_len += match_len
        i += 1

    return tot_matches_len / len(vec1)


def get_sorted_indices(vector):
    sorted_scores = [(j, vector[j]) for j in range(len(vector))]
    sorted_scores.sort(key=lambda tup: -tup[1])

    # Get the indices of the sorted pas, then a list of the sorted realized pas.
    return [sorted_score[0] for sorted_score in sorted_scores]


def edit_dist(vec1, vec2):
    dim = len(vec1)
    d = np.zeros((dim, dim))
    for i in range(dim):
        d[i, 0] = i
        d[0, i] = i

    for i in range(1, dim):
        for j in range(1, dim):
            if vec1[i] == vec2[j]:
                subst_cost = 0
            else:
                subst_cost = 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + subst_cost)

    return (dim - d[dim-1, dim-1]) / dim


doc_matrix, _, score_matrix = get_matrices(-1, "bestN", False, (0.3, 0.7))

test_docs = doc_matrix[422:, :, :]
test_scores = score_matrix[422:, :]

model = build_model(385, 134, "mse", 10, "hard_sigmoid")
predicted_scores = model.predict(test_docs, batch_size=1)

pre_perc = 0
pre_perc_edit = 0
for i in range(test_docs.shape[0]):
    test_doc = test_docs[i]
    unpadded_dim = (test_doc[~np.all(test_doc == 0, axis=1)]).shape[0]
    unpadded_dim = int(unpadded_dim / 4)
    pre_perc += dist(get_sorted_indices(predicted_scores[i])[:unpadded_dim],
                     get_sorted_indices(test_scores[i])[:unpadded_dim])
    pre_perc_edit += edit_dist(get_sorted_indices(predicted_scores[i])[:unpadded_dim],
                               get_sorted_indices(test_scores[i])[:unpadded_dim])

pre_perc /= test_docs.shape[0]
pre_perc_edit /= test_docs.shape[0]

doc_matrix = doc_matrix[:422, :, :]
score_matrix = score_matrix[:422, :]
train_model(model, "test", doc_matrix, score_matrix, 0, 1,
            batch_size=1, val_size=50, save_model=False)

predicted_scores = model.predict(test_docs, batch_size=1)
post_perc = 0
post_perc_edit = 0
for i in range(test_docs.shape[0]):
    test_doc = test_docs[i]
    unpadded_dim = (test_doc[~np.all(test_doc == 0, axis=1)]).shape[0]
    unpadded_dim = int(unpadded_dim / 4)
    post_perc += dist(get_sorted_indices(predicted_scores[i])[:unpadded_dim],
                      get_sorted_indices(test_scores[i])[:unpadded_dim])
    post_perc_edit += edit_dist(get_sorted_indices(predicted_scores[i])[:unpadded_dim],
                                get_sorted_indices(test_scores[i])[:unpadded_dim])

post_perc /= test_docs.shape[0]
post_perc_edit /= test_docs.shape[0]

print(pre_perc)
print(pre_perc_edit)

print(post_perc)
print(post_perc_edit)

#"""

"""
def equiv_test(set0, set1):
    pooled_var = (np.var(set0) * (len(set0) - 1) + np.var(set1) * (len(set1) - 1)) / (len(set0) + len(set1) - 2)
    print(pooled_var)
    return (abs(np.mean(set0) - np.mean(set1)) - 0.02) / sqrt(pow(pooled_var, 2) * (1 / len(set0) + 1 / len(set1)))

with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section0_take3_rc_list.dat", "rb") as f:
    tk0 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section50_take0_rc_list.dat", "rb") as f:
    tk1 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section100_take1_rc_list.dat", "rb") as f:
    tk2 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section150_take2_rc_list.dat", "rb") as f:
    tk3 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section200_take4_rc_list.dat", "rb") as f:
    tk4 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section250_take3_rc_list.dat", "rb") as f:
    tk5 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section300_take1_rc_list.dat", "rb") as f:
    tk6 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section350_take1_rc_list.dat", "rb") as f:
    tk7 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section400_take4_rc_list.dat", "rb") as f:
    tk8 = pickle.load(f)
with open(os.getcwd() + "/results/recall_lists/xval_duc_bin_(0.8, 0.2)_section450_take1_rc_list.dat", "rb") as f:
    tk9 = pickle.load(f)



print(np.var(tk0))
print(np.var(tk1))
print(np.var(tk2))
print(np.var(tk3))
print(np.var(tk4))
print(np.var(tk5))
print(np.var(tk6))
print(np.var(tk7))
print(np.var(tk8))
print(np.var(tk9))
print(np.mean(tk0))
print(np.mean(tk1))

print(equiv_test(tk0, tk1))
print(equiv_test(tk1, tk2))

print(stats.f_oneway(tk0, tk1, tk2, tk3, tk4, tk5, tk6, tk7, tk8, tk9))
"""

"""
for i in range(0, 35):
    print("matrices {}".format(i))
    for scores in ("non_bin", "bin", "bestN"):
        print("scores: {} {}".format(i, scores))
        store_score_matrices(i, scores, True)
"""

"""     TESTING WEIGHTED PAS METHOD (SIMPLE)
weights_list = [#  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                #  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                # [0.2, 0.1, 0.3, 0.1, 0.1, 0.2],
                 [0.3, 0.05, 0.3, 0.1, 0.05, 0.2],
                # [0.1, 0.05, 0.3, 0.1, 0.15, 0.2]
                ]

duc_dataset = True
extractive = True
ds_threshold = 0.15

if duc_dataset:
    training_no = 422
    batches = 0
    duc_index = -1
    dataset = "DUC"
else:
    training_no = 832  # includes validation.
    batches = 35
    duc_index = 0
    dataset = "NYT"

rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                "rouge_2_precision": 0, "rouge_2_f_score": 0}


for weights in weights_list:
    for index in range(duc_index, batches):
        if index == 34:
            training_no = 685

        docs_pas_lists, refs_pas_lists = get_pas_lists(index)
        docs_pas_lists = docs_pas_lists[training_no:]
        refs = get_sources_from_pas_lists(refs_pas_lists[training_no:])

        if extractive:
            docs_sent_lists = [[pas.sentence for pas in pas_list] for pas_list in docs_pas_lists]
            sent_matrix, _, _ = get_matrices(index, "bin", True, (0.3, 0.7))
            vectors_lists = sent_matrix[training_no:, :, :6]
            score = dataset_rouge_scores_weighted_extractive(docs_sent_lists, vectors_lists, refs, weights)
        else:
            score = dataset_rouge_scores_weighted(docs_pas_lists, refs, weights)

        rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
        rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
        rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
        rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
        rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
        rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

    for k in rouge_scores.keys():
        rouge_scores[k] /= batches - duc_index

    with open(os.getcwd() + "/results/results.txt", "a") as res_file:
        print(dataset + " w no ds" + str(weights), file=res_file)
        print(rouge_scores, file=res_file)
        print("=================================================", file=res_file)
"""

""" CHECKING DIRECT SPEECH
dataset_len = 0
ds_indices = []

duc = False
if duc:
    duc_index = -1
    batches = 0
    training_no = 422
else:
    duc_index = 0
    batches = 2
    training_no = 832

for index in range(duc_index, batches):
    if index == 34:
        training_no = 685
    print("Processing batch: " + str(index))
    docs_pas_lists, _ = get_pas_lists(index=index)
    docs_pas_lists = docs_pas_lists[training_no:]
    #docs = get_sources_from_pas_lists(docs_pas_lists)

    for pas_list in docs_pas_lists:
        doc_index = docs_pas_lists.index(pas_list)
        print("batch {}. processing doc {}/{}".format(index, doc_index, len(docs_pas_lists)))
        size = 0
        ds_size = 0
        used_sentences = []

        if direct_speech_ratio_pas(pas_list) > 0.15:
            ds_indices.append(doc_index)
    dataset_len += len(docs_pas_lists)

print(len(ds_indices) / dataset_len)
"""

"""        SUMM RATIO

doc_len = 0
summ_len = 0
duc = False

if duc:
    duc_index = -1
    batches = 0
else:
    duc_index = 0
    batches = 35

for index in range(duc_index, batches):
    print("Processing batch: " + str(index))
    docs_pas_lists, refs_pas_lists = get_pas_lists(index=index)
    docs = get_sources_from_pas_lists(docs_pas_lists)
    refs = get_sources_from_pas_lists(refs_pas_lists)

    for doc in docs:
        doc_len += len(doc.split())
    for ref in refs:
        summ_len += len(ref.split())

print((doc_len - summ_len) / doc_len)

"""
"""        COMPUTING MAXIMUM SCORES (PER SCORING METHOD)
duc_dataset = False
ds_threshold = 0.15

weights_list = [(0.0, 1.0),
                #(0.1, 0.9),
                #(0.2, 0.8), (0.3, 0.7),
                #(0.4, 0.6), (0.5, 0.5),
                #(0.6, 0.4), (0.7, 0.3),
                #(0.8, 0.2), (0.9, 0.1),
                #(1.0, 0.0)
                 ]

if duc_dataset:
    training_no = 422
    batches = 0
    duc_index = -1
else:
    training_no = 832  # includes validation.
    batches = 35
    duc_index = 0

for weights in weights_list:
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}

    docs_no = 0
    for k in range(duc_index, batches):
        doc_matrix, ref_matrix, score_matrix = get_matrices(k, "bestN", 0, weights)
        docs_pas_lists, refs_pas_lists = get_pas_lists(k)
        refs = get_sources_from_pas_lists(refs_pas_lists)

        if k == 34:
            training_no = 685

        docs_pas_lists = docs_pas_lists[training_no:]
        doc_matrix = doc_matrix[training_no:, :, :]
        score_matrix = score_matrix[training_no:, :]
        refs = refs[training_no:]

        recall_scores_list = []
        summaries = []

        max_sent_no = doc_matrix.shape[1]

        for i in range(len(docs_pas_lists)):
            if direct_speech_ratio_pas(docs_pas_lists[i]) < ds_threshold:
                docs_no += 1
                pas_list = docs_pas_lists[i]
                pas_no = len(pas_list)
                sent_vec_len = len(pas_list[0].vector) + len(pas_list[0].embeddings)

                pred_scores = score_matrix[i, :]
                scores = pred_scores[:pas_no]

                summary = generate_summary(pas_list, scores, summ_len=len(refs[i].split()))

                score = document_rouge_scores(summary, refs[i])

                summaries.append(summary)
                recall_scores_list.append(score["rouge_1_recall"])

                rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
                rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
                rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
                rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
                rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
                rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            else:
                print("HELLO THERE")

        sample_summaries("maximum_scores_DUC_non_bin" + str(weights),
                         docs_pas_lists,
                         refs,
                         summaries,
                         recall_scores_list,
                         batch=k)

    for k in rouge_scores.keys():
        rouge_scores[k] /= docs_no

    with open(os.getcwd() + "/results/results.txt", "a") as res_file:
        print("maximum score NYT ds bestN" + str(weights), file=res_file)
        print(rouge_scores, file=res_file)
        print("=================================================", file=res_file)
"""

"""
docs, _ = get_pas_lists(-1)
for pas_list in docs[494:495]:
    pas_list = pas_list[11:16]
    for pas in pas_list:
        print(pas.realized_pas)
#    print(docs.index(pas_list))
    resolve_anaphora_pas_list(pas_list)
    print("___________________________________________")
    for pas in pas_list:
        print(pas.realized_pas)
"""

"""
text = '"The nomination of John Tower to be secretary of defense is not confirmed Quayle intoned after the vote.\n' \
       'Nancy Kassebaum provided the biggest surprise of the final hours of debate when she became the sole ' \
       'Republican to declare her opposition.\n' \
        'she became the sole Republican to declare her opposition.\n' \
        'the sole Republican declare her opposition.\n' \
        'She said her "most serious concerns"' \
       "were over Tower's activities as a defense consultant after serving as an arms control negotiator.\n"
text = "It is not that we have a short space of time, but that we waste much of it"
print(text)
for sent in resolve_anaphora(tokens(text)):
    print(sent)

print(text)
for sent in resolve_anaphora(tokens(text)):
    print(sent)
"""

"""   MASS TRAINING

dataset = "duc"
losses = ["mse"]
activations = ["hard_sigmoid"]
scores = [2]
dense_layers = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
epochs = [1]
batch_sizes = [1]
name = 300

for loss in losses:
    for score in scores:
        for epoch in epochs:
            for activation in activations:
                for dense_layer in dense_layers:
                    for batch_size in batch_sizes:
                        train(str(name), loss, dense_layer, activation, batch_size, epoch, score, dataset, (0.3, 0.7))
                        name += 1

"""

""" DUMMY MODEL
model = build_model(4, 2, "matching_ones", 1, "sigmoid")

doc_mat = np.array([
                        [
                            [1, 1], [3, 5], [2, 3], [6, 2]
                        ],
                        [
                            [3, 3], [5, 1], [9, 3], [0, 0]
                        ],
                        [
                            [0, 1], [3, 2], [1, 5], [0, 0]
                        ]
                    ])

score_mat = np.array([[1, 1, 0, 1],
                      [0, 1, 1, 0],
                      [0, 0, 1, 0]])


print(model.predict([[[[0, 1], [3, 2], [1, 5], [0, 0]]]]))

train_model(model, "loss_test", doc_mat, score_mat, 0, 1, val_size=1)

print(model.predict([[[[0, 1], [3, 2], [1, 5], [0, 0]]]]))
"""

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
