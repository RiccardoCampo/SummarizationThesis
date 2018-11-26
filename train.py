import os
import sys
from dataset_scores import get_matrices
from deep_model import build_model, train_model


def train(series_name, loss, dense_layers, out_act, batch_size, epochs, scores_type, dataset, extractive, weights=None):
    """
    Train a model with the specified parameters.

    :param series_name: name of the model or training series number.
    :param loss: loss function.
    :param dense_layers: number of dense layer after the BLSTM.
    :param out_act: output activation.
    :param batch_size: batch size.
    :param epochs: epochs.
    :param scores_type: "bin", "non_bin" or "bestN"
    :param dataset: dataset with which the model will be trained.
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

    last_index_size = 685       # Size of the last batch training portion for nyt dataset.

    # Indices varies based on the dataset.
    if dataset == "nyt":
        batches = 35
        train_size = 666
        val_size = 166
        doc_size = 300      # Max size of the matrices in case of nyt dataset.
        duc_index = 0       # Used to set the parameter "index" to -1 when using DUC, to get duc matrices and scores.
    else:
        batches = 0
        train_size = 372
        val_size = 50
        doc_size = 385
        #duc_index = -1
        duc_index = 0

    if extractive:
        doc_size = 200

    vector_size = 134

    for weights in weights_list:
        model_name = dataset + "_" + series_name + "_" + loss + "_dense" + str(dense_layers) + "_" + out_act + "_bs" + \
                     str(batch_size) + "_ep" + str(epochs) + "_scores" + str(scores_type) + "_" + str(weights)
        save_model = False              # Saves the model only when the training process is complete (last batch).

        training_no = train_size + val_size

        model = build_model(doc_size, vector_size, loss, dense_layers, out_act)
        for index in range(duc_index, batches):
            if index == 34:
                training_no = last_index_size

            doc_matrix, _, score_matrix = get_matrices(index, scores_type, extractive, weights)
            doc_matrix = doc_matrix[:training_no, :, :]
            score_matrix = score_matrix[:training_no, :]

            if index == batches - 1:
                save_model = True

            # Compute the initial epoch to obtain a continuous graph.
            init_ep = (index if index >= 0 else 0) * epochs

            print(weights)
            print("batch: " + str(index))
            train_model(model, model_name, doc_matrix, score_matrix, init_ep, init_ep + epochs,
                        batch_size=batch_size, val_size=val_size, save_model=save_model)


if __name__ == "__main__":
    if str(sys.argv[1]) == "--help" or str(sys.argv[1]) == "-h":
        print("Usage:")
        print("test.py model_name loss_function #dense_layers output_activation batch_size "
              "epochs scores train_dataset extractive [weight1 weight2]")
        print("* scores can be non_bin, bin, bestN")
        print("* dataset can be either duc or nyt")
        print("* extractive can be 0 or 1")
        print("* if weights are not specified it will look for every weight")
    else:
        name = str(sys.argv[1])             # Model name.
        ls = str(sys.argv[2])               # Loss function.
        dn = int(sys.argv[3])               # Number of dense layers.
        oa = str(sys.argv[4])               # Output activation.
        bs = int(sys.argv[5])               # Batch size.
        ep = int(sys.argv[6])               # Epochs.
        sc = int(sys.argv[7])               # Scores: 0 non binary, 1 binary, 2 closest binary.
        dset = str(sys.argv[8])             # Dataset
        extr = bool(int(sys.argv[9]))       # Extractive: 1 if extractive summarization.
        print(extr)
        if len(sys.argv) > 10:
            w1 = float(sys.argv[10])         # Weight 1.
            w2 = float(sys.argv[11])        # Weight 2
            train(name, ls, dn, oa, bs, ep, sc, dset, extr, (w1, w2))
        else:
            train(name, ls, dn, oa, bs, ep, sc, dset, extr)
