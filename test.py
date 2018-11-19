import os
import pickle
import sys
import numpy as np

from dataset_scores import get_matrices
from dataset_text import get_pas_lists, get_duc
from summarization import dataset_rouge_scores_deep, dataset_rouge_scores_extract
from utils import get_sources_from_pas_lists


def test(series_name, test_dataset, train_dataset, extractive, weights=None):
    """
    Compute ROUGE score of the specified model using the specified dataset.

    :param series_name: name of the model or training series number.
    :param test_dataset: dataset with which test the model.
    :param train_dataset: dataset with which the model was trained.
    :param extractive: whether it is extractive summarization or not.
    :param weights: a tuple of two weights to average 0/1 clustering and N clusters.
    """
    # If the weights are not specified all of them are used.
    if weights:
        weights_list = [weights]
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]

    # Indices varies based on the dataset.
    last_index_size = 685
    if test_dataset == "nyt":
        batches = 35
        duc_index = 0       # Used to set the parameter "index" to -1 when using DUC, to get duc matrices and scores.
        training_no = 832   # Includes validation.
        max_doc_len = 300   # Max size of the matrices in case of nyt dataset.
    else:
        batches = 0         # With duc it will only consider the value -1 (duc matrices using get_matrices).
        duc_index = -1
        training_no = 422
        max_doc_len = 385

    for weights in weights_list:
        model_name = series_name + "_" + str(weights)
        rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                        "rouge_2_precision": 0, "rouge_2_f_score": 0}
        recall_list = []                            # Storing the recall for each document.
        for index in range(duc_index, batches):
            if index == 34:
                training_no = last_index_size

            doc_matrix, _, _ = get_matrices(index, "bin", extractive, weights)
            doc_matrix = doc_matrix[training_no:, :, :]

            if extractive:
                docs, refs, _ = get_duc()
                docs = docs[training_no:]
                refs = refs[training_no:]

                score, recall_list_part = dataset_rouge_scores_extract(model_name, docs, doc_matrix, refs,
                                                                       dynamic_summ_len=True, batch=index, rem_ds=True)
            else:
                if test_dataset != train_dataset:
                    if test_dataset == "nyt":                           # Test DUC with NYT model.
                        doc_matrix = doc_matrix[training_no:, :max_doc_len, :]
                    else:                                               # Test NYT with DUC model.
                        extended_doc_matrix = np.zeros((doc_matrix.shape[0], max_doc_len, doc_matrix.shape[2]))
                        extended_doc_matrix[:doc_matrix.shape[0],
                                            :doc_matrix.shape[1], :doc_matrix.shape[2]] = doc_matrix
                        doc_matrix = extended_doc_matrix

                docs_pas_lists, refs_pas_lists = get_pas_lists(index)
                refs = get_sources_from_pas_lists(refs_pas_lists)
                docs_pas_lists = docs_pas_lists[training_no:]
                refs = refs[training_no:]

                score, recall_list_part = dataset_rouge_scores_deep(model_name, docs_pas_lists, doc_matrix, refs,
                                                                    dynamic_summ_len=True, batch=index, rem_ds=True)

            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            recall_list.extend(recall_list_part)

        # Averaging the scores wrt the number of batches.
        for k in rouge_scores.keys():
            rouge_scores[k] /= batches - duc_index  # if duc then /1 else /35

        # Get validation accuracy histories.
        with open(os.getcwd() + "/results/histories/" + model_name + ".hst", "rb") as file:
            history = pickle.load(file)
        val_acc = history['val_acc']

        with open(os.getcwd() + "/results/results.txt", "a") as res_file:
            print(model_name + "   val_acc: " + str(val_acc[-1]) +
                  "   tested on: " + test_dataset + " with #  of docs: " + str(len(recall_list)), file=res_file)
            print(rouge_scores, file=res_file)
            print("=================================================", file=res_file)

        with open(os.getcwd() + "/results/recall_lists/" + model_name + "_rc_list.dat", "wb") as list_file:
            pickle.dump(recall_list, list_file)


if __name__ == "__main__":
    if str(sys.argv[1]) == "--help" or str(sys.argv[1]) == "-h":
        print("Usage:")
        print("test.py model_name test_dataset train_dataset extractive [weight1 weight2]")
        print("* dataset can be either duc or nyt")
        print("* extractive can be 0 or 1")
        print("* if weights are not specified it will look for every weight")
    else:
        name = str(sys.argv[1])
        tst_set = str(sys.argv[2])
        trn_set = str(sys.argv[3])
        extr = bool(int(sys.argv[4]))
        if len(sys.argv) > 5:
            w1 = float(sys.argv[5])
            w2 = float(sys.argv[6])
            test(name, tst_set, trn_set, extr, (w1, w2))
        else:
            test(name, tst_set, trn_set, extr)
