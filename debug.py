import os
import pickle
import re
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import logging
import keras

from dataset import get_matrices, get_duc, shuffle_data, get_nyt, \
    store_pas_nyt_dataset, compute_idfs, store_matrices, get_nyt_pas_lists, arrange_nyt_pas_lists, get_refs_from_pas
from loss_testing import summary_clustering_score, summary_clustering_score_2
from summarization import testing, testing_weighted, find_redundant_pas, rouge_score, build_model, train_model, best_pas
from utils import sentence_embeddings, plot_history

_duc_path_ = os.getcwd() + "/dataset/duc_source"
_nyt_path_ = "D:/Datasets/nyt_corpus/data"

if os.name == "posix":
    _result_path_ = "/home/arcslab/Documents/Riccardo_Campo/results/results.txt"
else:
    _result_path_ = "C:/Users/Riccardo/Desktop/temp_results/results.txt"


plot_history("nyt_001_10_4_(0.0, 1.0)")
plot_history("nyt_001_10_4_(0.1, 0.9)")
plot_history("nyt_001_10_4_(0.2, 0.8)")
plot_history("nyt_001_10_4_(0.3, 0.7)")
plot_history("nyt_001_10_4_(0.4, 0.6)")
plot_history("nyt_001_10_4_(0.5, 0.5)")
plot_history("nyt_001_10_4_(0.6, 0.4)")
plot_history("nyt_001_10_4_(0.7, 0.3)")
plot_history("nyt_001_10_4_(0.8, 0.2)")
plot_history("nyt_001_10_4_(0.9, 0.1)")
plot_history("nyt_001_10_4_(1.0, 0.0)")



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


"""        COMPUTING MAXIMUM SCORES (PER SCORING METHOD)    DUC
weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                (0.4, 0.6), (0.5, 0.5),(0.6, 0.4), (0.7, 0.3),
                (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]

for weights in weights_list:
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}
    batches = 7

    for k in range(batches):
        doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, index=k)
        docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=k)
        refs = get_refs_from_pas(refs_pas_lists)
        # _, refs, _ = get_duc(_duc_path_)       DUC

        training_no = 666       # includes validation.

        docs_pas_lists = docs_pas_lists[training_no:]
        doc_matrix = doc_matrix[training_no:, :, :]
        score_matrix = score_matrix[training_no:, :]
        refs = refs[training_no:]

        max_sent_no = doc_matrix.shape[1]

        for i in range(len(docs_pas_lists)):
           # print(weights)
           # print(k)
           # print("Processing doc:" + str(i) + "/" + str(len(docs_pas_lists)))
            pas_list = docs_pas_lists[i]
            pas_no = len(pas_list)
            sent_vec_len = len(pas_list[0].vector) + len(pas_list[0].embeddings)

            pred_scores = score_matrix[i, :]
            scores = pred_scores[:pas_no]
            sorted_scores = [(j, scores[j]) for j in range(len(scores))]
            sorted_scores.sort(key=lambda tup: -tup[1])

            sorted_indices = [sorted_score[0] for sorted_score in sorted_scores]
            sorted_realized_pas = [pas_list[index].realized_pas for index in sorted_indices]
            best_pas_list = []
            best_indices_list = []
            size = 0
            j = 0
            while size < 100 and j < pas_no:
                redundant_pas = find_redundant_pas(best_pas_list, sorted_realized_pas[j])
                if redundant_pas is None:
                    size += len(sorted_realized_pas[j].split())
                    if size < 100:
                        best_pas_list.append(sorted_realized_pas[j])
                        best_indices_list.append(sorted_indices[j])
                else:
                    if redundant_pas in best_pas_list:
                        if size - len(redundant_pas) + len(sorted_realized_pas[j]) < 100:
                            size = size - len(redundant_pas) + len(sorted_realized_pas[j])
                            best_pas_list[best_pas_list.index(redundant_pas)] = sorted_realized_pas[j]
                            best_indices_list[best_pas_list.index(redundant_pas)] = sorted_indices[j]
                j += 1

            best_indices_list.sort()

            summary = ""
            for index in best_indices_list:
                summary += pas_list[index].realized_pas + ".\n"

            score = rouge_score([summary], [refs[i]])
            #print(score["rouge_1_recall"])
            if score["rouge_1_recall"] < 0.15:
                print("===================================")
                print(score["rouge_1_recall"])
                print(k)
                print(i)
                print("===================================")
"""
"""
            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

    for k in rouge_scores.keys():
        rouge_scores[k] /= 334 * batches

    with open(_result_path_, "a") as res_file:
        print("maximum score" + str(weights), file=res_file)
        print(rouge_scores, file=res_file)
        print("=================================================", file=res_file)

"""

"""        TESTING & TRAINING DUC
tst = False
binary = False
for i in range(0, 10):
    weights_list = [#(0.4, 0.6),
                    #(0.5, 0.5),
                    #(0.6, 0.4),
                    #(0.3, 0.7)
                    (0.2, 0.8),
                    #(0.1, 0.9),
                    #(0.0, 1.0)
                    ]
    for weights in weights_list:
        store_duc_matrices(weights, binary_scores=binary)
        doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, binary=binary)
        docs_pas_lists, _ = get_pas_lists()
        _, refs, _ = get_duc(_duc_path_)

        training_no = 348       # includes validation.

        model_name = "correct_blstm" + str(weights) + str(i)
        if not tst:
            training(doc_matrix[:training_no, :, :], score_matrix[:training_no, :], model_name, epochs=1)
        print(model_name)

        if tst:
            score = testing(model_name,
                            docs_pas_lists[training_no:],
                            doc_matrix[training_no:, :, :],
                            refs[training_no:])
            print(score)
            with open(_result_path_, "a") as res_file:
                print(model_name, file=res_file)
                print(score, file=res_file)
                print("=================================================", file=res_file)
"""

"""     DUMMY DATA  
#doc_matrix, ref_matrix, score_matrix = get_matrices(weights=(0.0, 1.0), binary=True)
#store_duc_matrices((0.0, 1.0), binary_scores=True)
#print(doc_matrix[0, 0, :])

docs = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
                 [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.5, 1.0, 1.5, 2.0]],
                 [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                 [[3.0, 6.0, 9.0, 12.0], [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]])
scores = np.array([[1.0, 2.0, 0.0],
                   [1.0, 2.0, 0.5],
                   [1.0, 0.0, 0.0],
                   [3.0, 1.0, 0.0]])
training(docs, scores, "DUMMY", epochs=10)
"""

"""        TESTING & TRAINING NYT
tst = False
trn = True
binary = False
weights_list = [(0.0, 1.0),
                #(0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                #(0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                #(0.8, 0.2), (0.9, 0.1), (1.0, 0.0)
                 ]
batches = 3
training_no = 666                   # includes validation.
doc_size = 300
vector_size = 134

for weights in weights_list:
    model_name = "nyt_first" + str(weights)
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
            train_model(model, model_name, doc_matrix[:training_no, :, :], score_matrix[:training_no, :], epochs=4, batch_size=50, save_model=save_model)

    if tst:
        rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                        "rouge_2_precision": 0, "rouge_2_f_score": 0}
        for index in range(batches):
            doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, binary=binary, index=index)
            docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index)
            refs = get_refs_from_pas(refs_pas_lists)
            score = testing(model_name,
                            docs_pas_lists[training_no:],
                            doc_matrix[training_no:, :, :],
                            refs[training_no:])

            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

        for k in rouge_scores.keys():
            rouge_scores[k] /= batches

        with open(_result_path_, "a") as res_file:
            print(model_name, file=res_file)
            print(rouge_scores, file=res_file)
            print("=================================================", file=res_file)

"""

#plot_history("nyt_first(0.0, 1.0)")


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