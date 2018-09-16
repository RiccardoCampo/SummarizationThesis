import pickle
import sys
import os

from dataset import get_matrices, get_nyt_pas_lists
from summarization import testing
from utils import sample_summaries, get_sources_from_pas_lists, result_path



def test(series_name):
    binary = False
    weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                    (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                    (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    batches = 15
    training_no = 666  # includes validation.

    for weights in weights_list:
        model_name = series_name + "_" + str(weights)
        rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                        "rouge_2_precision": 0, "rouge_2_f_score": 0}
        recall_list = []
        for index in range(batches):
            doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, binary=binary, index=index)
            docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index)
            refs = get_sources_from_pas_lists(refs_pas_lists)
            score, recall_list_part = testing(model_name,
                                              docs_pas_lists[training_no:],
                                              doc_matrix[training_no:, :, :],
                                              refs[training_no:])

            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            recall_list.extend(recall_list_part)

            sample_summaries(model_name, docs_pas_lists, refs, recall_list_part, batch=index)

        for k in rouge_scores.keys():
            rouge_scores[k] /= batches

        with open(result_path + "results.txt", "a") as res_file:
            print(model_name, file=res_file)
            print(rouge_scores, file=res_file)
            print("=================================================", file=res_file)

        with open(result_path + "recall_lists/" + model_name + "_rc_list.dat", "wb") as list_file:
            pickle.dump(recall_list, list_file)


if __name__ == "__main__":
    name = str(sys.argv[1])
    test(name)
