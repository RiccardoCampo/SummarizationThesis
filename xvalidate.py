import os
import pickle
import sys
import numpy as np

from dataset_scores import get_matrices
from dataset_text import get_pas_lists
from deep_model import build_model, train_model
from summarization import dataset_rouge_scores_deep
from utils import get_sources_from_pas_lists


def x_validate(dataset, scores_type, weights):

    if dataset == "nyt":
        cutting_index = 5000
        section_dim = 50
        doc_size = 300  # Max size of the matrices in case of nyt dataset.
        batch_size = 20
        doc_matrix, _, score_matrix = get_matrices(0, scores_type, False, weights)
        doc_matrix = doc_matrix[:150, :, :]
        score_matrix = score_matrix[:150, :]
        docs_pas_lists, refs_pas_lists = get_pas_lists(0)
        docs_pas_lists = docs_pas_lists[:150]
        refs_pas_lists = refs_pas_lists[:150]
        for i in range(1, 34):
            doc_sub_matrix, _, score_sub_matrix = get_matrices(i, scores_type, False, weights)
            doc_matrix = np.append(doc_matrix, doc_sub_matrix[:150, :, :])
            score_matrix = np.append(score_matrix, score_sub_matrix[:150, :])

            docs_pas_lists_part, refs_pas_lists_part = get_pas_lists(i)
            docs_pas_lists.extend(docs_pas_lists_part[:150])
            refs_pas_lists.extend(refs_pas_lists_part[:150])
    else:
        cutting_index = 500
        section_dim = 50
        doc_size = 385
        batch_size = 1
        doc_matrix, _, score_matrix = get_matrices(-1, scores_type, False, weights)
        docs_pas_lists, refs_pas_lists = get_pas_lists(-1)


    indices = np.random.permutation(doc_matrix.shape[0])
    doc_matrix = doc_matrix[indices, :, :]
    score_matrix = score_matrix[indices, :]
    docs_pas_lists = [docs_pas_lists[index] for index in indices]
    refs_pas_lists = [refs_pas_lists[index] for index in indices]

    doc_matrix = doc_matrix[:cutting_index, :, :]
    score_matrix = score_matrix[:cutting_index, :]
    docs_pas_lists = docs_pas_lists[:cutting_index]
    refs_pas_lists = refs_pas_lists[:cutting_index]
    refs = get_sources_from_pas_lists(refs_pas_lists)

    for i in range(0, cutting_index, section_dim):
        for j in range(5):
            print("Processing batch" + str(i))
            model_name = "xval_{}_{}_{}_section{}_take{}".format(dataset, scores_type, weights, i, j)
            train_docs = np.concatenate([doc_matrix[:i, :, :], doc_matrix[i+section_dim:, :, :]])
            train_scores = np.concatenate([score_matrix[:i, :], score_matrix[i+section_dim:, :]])
            test_doc = doc_matrix[i:i+section_dim, :, :]
            section_docs_pas_lists = docs_pas_lists[i:i+section_dim]
            section_refs = refs[i:i+section_dim]

            model = build_model(doc_size, 134, "mse", 10, "hard_sigmoid")
            train_model(model, model_name, train_docs, train_scores, 0, 1,
                        batch_size=batch_size, val_size=0, save_model=True)

            rouge_scores, recall_list = dataset_rouge_scores_deep(model_name, section_docs_pas_lists, test_doc,
                                                                  section_refs, dynamic_summ_len=True,
                                                                  batch=-1, rem_ds=False)

            with open(os.getcwd() + "/results/results.txt", "a") as res_file:
                print(model_name +
                      "   tested on: " + dataset + " with #  of docs: " + str(len(recall_list)), file=res_file)
                print(rouge_scores, file=res_file)
                print("=================================================", file=res_file)

            with open(os.getcwd() + "/results/recall_lists/" + model_name + "_rc_list.dat", "wb") as list_file:
                pickle.dump(recall_list, list_file)


if __name__ == "__main__":
    dset = str(sys.argv[1])
    s_type = str(sys.argv[2])
    w1 = float(sys.argv[3])
    w2 = float(sys.argv[4])

    x_validate(dset, s_type, (w1, w2))