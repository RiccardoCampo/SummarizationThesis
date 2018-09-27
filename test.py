import os
import pickle
import sys

from dataset import get_matrices, get_pas_lists
from summarization import testing
from utils import get_sources_from_pas_lists


def test(series_name, dataset, weights=None):
    if weights:
        weights_list = [weights]
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]

    last_index_size = 685
    if dataset == "nyt":
        batches = 35
        duc_index = 0
        training_no = 666  # includes validation.
    else:
        batches = 0
        duc_index = -1
        training_no = 422  # includes validation.

    for weights in weights_list:
        model_name = series_name + "_" + str(weights)
        rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                        "rouge_2_precision": 0, "rouge_2_f_score": 0}
        recall_list = []
        for index in range(duc_index, batches):
            if index == 34:
                training_no = last_index_size
            doc_matrix, _, _ = get_matrices(weights=weights, binary=False, index=index)
            docs_pas_lists, refs_pas_lists = get_pas_lists(index)
            refs = get_sources_from_pas_lists(refs_pas_lists)

            docs_pas_lists = docs_pas_lists[training_no:]
            doc_matrix = doc_matrix[training_no:, :300, :]
            refs = refs[training_no:]

            score, recall_list_part = testing(model_name,
                                              docs_pas_lists,
                                              doc_matrix,
                                              refs,
                                              dynamic_summ_len=True,
                                              batch=index,
                                              rem_ds=True)

            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            recall_list.extend(recall_list_part)

        for k in rouge_scores.keys():
            rouge_scores[k] /= batches - duc_index  # if duc then /1 else /35

        with open(os.getcwd() + "/results/histories/" + model_name + ".hst", "rb") as file:
            history = pickle.load(file)
        # Get training and test loss histories
        val_acc = history['val_acc']

        with open(os.getcwd() + "/results/results.txt", "a") as res_file:
            print(model_name + "   val_acc: " + str(val_acc[-1]), file=res_file)
            print(rouge_scores, file=res_file)
            print("=================================================", file=res_file)

        with open(os.getcwd() + "/results/recall_lists/" + model_name + "_rc_list.dat", "wb") as list_file:
            pickle.dump(recall_list, list_file)


if __name__ == "__main__":
    name = str(sys.argv[1])
    dset = str(sys.argv[2])
    if len(sys.argv) > 3:
        w1 = float(sys.argv[3])
        w2 = float(sys.argv[4])
        test(name, dset, (w1, w2))
    else:
        test(name, dset)
