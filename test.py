import pickle
import sys
import numpy as np

from dataset import get_matrices, get_nyt_pas_lists
from summarization import testing
from utils import get_sources_from_pas_lists, result_path, direct_speech_ratio


def test(series_name, weights=None, ds_threshold=0.15):
    if weights:
        weights_list = [weights]
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    batches = 35
    training_no = 666  # includes validation.

    for weights in weights_list:
        model_name = series_name + "_" + str(weights)
        rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                        "rouge_2_precision": 0, "rouge_2_f_score": 0}
        recall_list = []
        for index in range(batches):
            doc_matrix, _, _ = get_matrices(weights=weights, binary=False, index=index)
            docs_pas_lists, refs_pas_lists = get_nyt_pas_lists(index)
            refs = get_sources_from_pas_lists(refs_pas_lists)

            docs_pas_lists = docs_pas_lists[training_no:]
            doc_matrix = doc_matrix[training_no:, :, :]
            refs = refs[training_no:]

            if ds_threshold > 0:
                bad_doc_indices = []
                for doc_pas_list in docs_pas_lists:
                    if direct_speech_ratio(doc_pas_list) > ds_threshold:
                        bad_doc_indices.append(docs_pas_lists.index(doc_pas_list))

                deleted_docs = 0
                for bad_doc_index in bad_doc_indices:
                    bad_doc_index -= deleted_docs
                    doc_matrix = np.delete(doc_matrix, bad_doc_index, 0)
                    del docs_pas_lists[bad_doc_index]
                    del refs[bad_doc_index]
                    deleted_docs += 1

            score, recall_list_part = testing(model_name,
                                              docs_pas_lists,
                                              doc_matrix,
                                              refs,
                                              dynamic_summ_len=True,
                                              batch=index)

            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            recall_list.extend(recall_list_part)

        for k in rouge_scores.keys():
            rouge_scores[k] /= batches

        with open(result_path + "histories/" + model_name + ".hst", "rb") as file:
            history = pickle.load(file)
        # Get training and test loss histories
        val_acc = history['val_acc']

        with open(result_path + "results.txt", "a") as res_file:
            print(model_name + "   val_acc: " + str(val_acc[-1]), file=res_file)
            print(rouge_scores, file=res_file)
            print("=================================================", file=res_file)

        with open(result_path + "recall_lists/" + model_name + "_rc_list.dat", "wb") as list_file:
            pickle.dump(recall_list, list_file)


if __name__ == "__main__":
    name = str(sys.argv[1])
    if len(sys.argv) > 2:
        w1 = float(sys.argv[2])
        w2 = float(sys.argv[3])
        test(name, (w1, w2))
    test(name)
