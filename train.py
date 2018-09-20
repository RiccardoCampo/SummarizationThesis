import sys
import numpy as np

from dataset import get_matrices, get_nyt_pas_lists
from summarization import build_model, train_model
from utils import plot_history, direct_speech_ratio


def train(series_name, batch_size, epochs, binary, weights=None, ds_threshold=0.15):
    if weights:
        weights_list = [weights]
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    batches = 35
    training_no = 666  # includes validation.
    doc_size = 300
    vector_size = 134

    for weights in weights_list:
        model_name = series_name + "_" + str(batch_size) + "_" + str(epochs) + "_" + str(weights)
        save_model = False

        model = build_model(doc_size, vector_size)
        for index in range(batches):
            doc_matrix, _, score_matrix = get_matrices(weights=weights, binary=binary, index=index)
            doc_matrix = doc_matrix[:training_no, :, :]
            score_matrix = score_matrix[:training_no, :]

            if ds_threshold > 0:
                docs_pas_lists, _ = get_nyt_pas_lists(index)
                docs_pas_lists = docs_pas_lists[:training_no]

                bad_doc_indices = []
                for doc_pas_list in docs_pas_lists:
                    if direct_speech_ratio(doc_pas_list) > ds_threshold:
                        bad_doc_indices.append(docs_pas_lists.index(doc_pas_list))

                deleted_docs = 0
                for bad_doc_index in bad_doc_indices:
                    bad_doc_index -= deleted_docs
                    np.delete(doc_matrix, bad_doc_index, 0)
                    np.delete(score_matrix, bad_doc_index, 0)
                    deleted_docs += 1

            if index == batches - 1:
                save_model = True

            print(weights)
            print("index: " + str(index))
            train_model(model, model_name, doc_matrix, score_matrix, epochs=epochs,
                        batch_size=batch_size, save_model=save_model)
            plot_history(model_name)


if __name__ == "__main__":
    name = str(sys.argv[1])
    bs = int(sys.argv[2])
    ep = int(sys.argv[3])
    bn = bool(int(sys.argv[4]))
    if len(sys.argv) > 6:
        w1 = float(sys.argv[5])
        w2 = float(sys.argv[6])
        train(name, bs, ep, bn, (w1, w2))
    else:
        train(name, bs, ep, bn)
