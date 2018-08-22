import os
import pickle
import re
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import logging

from dataset import get_matrices, get_duc, shuffle_data, get_nyt, \
    store_pas_nyt_dataset, compute_idfs, store_matrices, get_nyt_pas_lists, arrange_nyt_pas_lists, get_refs_from_pas
from summarization import training, testing, testing_weighted, find_redundant_pas, rouge_score
from utils import sentence_embeddings

_duc_path_ = os.getcwd() + "/dataset/duc_source"
_nyt_path_ = "D:/Datasets/nyt_corpus/data"

if os.name == "posix":
    _result_path_ = "/home/arcslab/Documents/Riccardo_Campo/results/results.txt"
else:
    _result_path_ = "C:/Users/Riccardo/Desktop/temp_results/results.txt"

#"""     TESTING WEIGHTED PAS METHOD (SIMPLE)
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
#"""


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
            with open("C:/Users/Riccardo/Desktop/temp_results/results.txt", "a") as res_file:
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

"""
docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index=0)
refs = get_refs_from_pas(refs_pas_lists)

training_no = 666  # includes validation.

docs_pas_lists = docs_pas_lists[training_no:]
refs_pas_lists = refs_pas_lists[training_no:]
refs = refs[training_no:]
index = 99
for pas in docs_pas_lists[index]:
    print(pas.realized_pas)

print("==============================")
for pas in refs_pas_lists[index]:
    print(pas.realized_pas)

print("==============================")
print(refs[index])
"""

#arrange_nyt_pas_lists()