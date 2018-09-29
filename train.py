import sys
from dataset import get_matrices
from summarization import build_model, train_model


def train(series_name, loss, seq_at_end, dense_layers, out_act, batch_size, epochs, scores, dataset, weights=None):
    model_name = dataset + "_" + series_name + "_" + loss + "_seq" + str(seq_at_end) + "_dense" + str(dense_layers) + \
                 "_" + out_act + "_bs" + str(batch_size) + "_ep" + str(epochs) + "_scores" + str(scores)

    if weights:
        weights_list = [weights]
    elif scores == 2:
        weights_list = [(0.0, 0.0)]
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]

    last_index_size = 685
    if dataset == "nyt":
        batches = 35
        train_size = 666
        val_size = 166
        doc_size = 300
        duc_index = 0
    else:
        batches = 0
        train_size = 372
        val_size = 50
        doc_size = 385
        duc_index = -1

    training_no = train_size + val_size
    vector_size = 134

    for weights in weights_list:
        model_name += "_" + str(weights)
        save_model = False

        model = build_model(doc_size, vector_size, loss, seq_at_end, dense_layers, out_act)
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
    sq = bool(int(sys.argv[3]))         # Sequence at the end.
    dn = int(sys.argv[4])               # Number of dense layers.
    oa = str(sys.argv[5])               # Output activation.
    bs = int(sys.argv[6])               # Batch size.
    ep = int(sys.argv[7])               # Epochs.
    sc = int(sys.argv[8])               # Scores: 0 non binary, 1 binary, 2 closest binary.
    dset = str(sys.argv[9])             # Dataset
    if len(sys.argv) > 10:
        w1 = float(sys.argv[9])         # Weight 1.
        w2 = float(sys.argv[10])        # Weight 2
        train(name, ls, sq, dn, oa, bs, ep, sc, dset, (w1, w2))
    else:
        train(name, ls, sq, dn, oa, bs, ep, sc, dset)
