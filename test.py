import sys
import os

from dataset import get_matrices, get_nyt_pas_lists, get_refs_from_pas
from summarization import testing

if os.name == "posix":
    _result_path_ = "/home/arcslab/Documents/Riccardo_Campo/results/results.txt"
else:
    _result_path_ = "C:/Users/Riccardo/Desktop/temp_results/results.txt"


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


if __name__ == "__main__":
    name = str(sys.argv[1])
    test(name)
