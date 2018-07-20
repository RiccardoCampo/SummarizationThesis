import os
import pickle
import re

from dataset import get_matrices, get_pas_lists, get_duc, store_duc_matrices, shuffle_data, get_nyt, \
    store_pas_nyt_dataset
from summarization import training, testing, testing_weighted

_duc_path_ = "C:/Users/Riccardo/Desktop/duc"
_nyt_path_ = "D:/Datasets/nyt_corpus/data"

#store_pas_nyt_dataset(_nyt_path_, 0, 5)

"""        TESTING
weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                (0.4, 0.6), (0.5, 0.5),(0.6, 0.4), (0.7, 0.3),
                (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
batch = 9
for weights in weights_list:
    model_name = "batch" + str(batch) + "/" + str(weights[0]) + "-" + str(weights[1])
    # model_name = str(weights[0]) + "-" + str(weights[1])

    doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, batch=batch)
    docs_pas_lists, _ = get_pas_lists(batch=batch)
    _, refs, _ = get_duc(_duc_path_, batch=batch)

    training_no = 348       # includes validation.

    score = testing(model_name,
                    docs_pas_lists[training_no:],
                    doc_matrix[training_no:, :, :],
                    score_matrix[training_no:, :],
                    refs[training_no:])
    with open("C:/Users/Riccardo/Desktop/temp_results/results.txt", "a") as res_file:
        print(model_name, file=res_file)
        print(score, file=res_file)
        print("=================================================", file=res_file)
"""


""" TRAINING
for i in range(3, 10):
    shuffle_data(i)
    weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                    (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                    (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    for weights in weights_list:
        model_name = "batch" + str(i) + "/" + str(weights[0]) + "-" + str(weights[1])
        doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, batch=i)
        docs_pas_lists, _ = get_pas_lists(batch=i)
        _, refs, _ = get_duc(_duc_path_, batch=i)

        training_no = 348       # includes validation.

        training(doc_matrix[:training_no, :, :], score_matrix[:training_no, :], model_name)
"""

"""     TESTING WEIGHTED PAS METHOD (SIMPLE)
#weights = [0.2, 0.1, 0.3, 0.1, 0.1, 0.2]
weights = [0.3, 0.05, 0.3, 0.1, 0.05, 0.2]
#weights = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
docs_pas_lists, _ = get_pas_lists()
_, refs, _ = get_duc(_duc_path_)

with open("C:/Users/Riccardo/Desktop/temp_results/results.txt", "a") as res_file:
    print(weights, file=res_file)
    print(testing_weighted(docs_pas_lists, refs, weights), file=res_file)
    print("=================================================", file=res_file)
"""