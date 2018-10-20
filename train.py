import sys
from dataset import get_matrices
from summarization import build_model, train_model


# Train a model with the specified parameters.
def train(series_name, loss, dense_layers, out_act, batch_size, epochs, scores, dataset, weights=None):
    # If the weights are not specified all of them are used.
    if weights:
        weights_list = [weights]
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]

    last_index_size = 685       # Size of the last batch training portion.

    # Indices varies based on the dataset.
    if dataset == "nyt":
        batches = 35
        train_size = 666
        val_size = 166
        doc_size = 300
        duc_index = 0       # Used to set the parameter "index" to -1 when using DUC, to get duc matrices and scores.
    else:
        batches = 0
        train_size = 372
        val_size = 50
        doc_size = 385
        duc_index = -1

    training_no = train_size + val_size
    vector_size = 134

    for weights in weights_list:
        model_name = dataset + "_" + series_name + "_" + loss + "_dense" + str(dense_layers) + "_" + out_act + "_bs" + \
                     str(batch_size) + "_ep" + str(epochs) + "_scores" + str(scores) + "_" + str(weights)
        save_model = False              # Saves the model only when the training process is complete (last batch).

        model = build_model(doc_size, vector_size, loss, dense_layers, out_act)
        for index in range(duc_index, batches):
            if index == 34:
                training_no = last_index_size

            doc_matrix, _, score_matrix = get_matrices(weights, scores, index=index)
            doc_matrix = doc_matrix[:training_no, :, :]
            score_matrix = score_matrix[:training_no, :]

            if index == batches - 1:
                save_model = True

            init_ep = (index if index >= 0 else 0) * epochs

            print(weights)
            print("batch: " + str(index))
            train_model(model, model_name, doc_matrix, score_matrix, init_ep, init_ep + epochs,
                        batch_size=batch_size, val_size=val_size, save_model=save_model)


if __name__ == "__main__":
    name = str(sys.argv[1])             # Model name.
    ls = str(sys.argv[2])               # Loss function.
    dn = int(sys.argv[3])               # Number of dense layers.
    oa = str(sys.argv[4])               # Output activation.
    bs = int(sys.argv[5])               # Batch size.
    ep = int(sys.argv[6])               # Epochs.
    sc = int(sys.argv[7])               # Scores: 0 non binary, 1 binary, 2 closest binary.
    dset = str(sys.argv[8])             # Dataset
    if len(sys.argv) > 9:
        w1 = float(sys.argv[9])         # Weight 1.
        w2 = float(sys.argv[10])        # Weight 2
        train(name, ls, dn, oa, bs, ep, sc, dset, (w1, w2))
    else:
        train(name, ls, dn, oa, bs, ep, sc, dset)
