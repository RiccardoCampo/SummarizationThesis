import sys
from dataset import get_matrices
from summarization import build_model, train_model


def train(series_name, batch_size, epochs, binary, dataset, weights=None):
    if weights:
        weights_list = [weights]
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    if dataset == "nyt":
        batches = 35
        train_size = 666
        val_size = 166
        doc_size = 300
        duc_index = 0
        #indices = permutation(35)
        #indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        #           10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        #           20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        #           30, 31, 32, 33, 34]
    else:
        batches = 0
        train_size = 372
        val_size = 50
        doc_size = 385
        duc_index = -1
        #indices = [-1]

    training_no = train_size + val_size
    vector_size = 134

    for weights in weights_list:
        if binary:
            model_name = series_name + "_" + str(batch_size) + "_" + str(epochs) + "_bin_" + str(weights)
        else:
            model_name = series_name + "_" + str(batch_size) + "_" + str(epochs) + "_" + str(weights)
        save_model = False

        model = build_model(doc_size, vector_size)
        for index in range(duc_index, batches):
        #for index in indices:
            doc_matrix, _, score_matrix = get_matrices(weights=weights, binary=binary, index=index)
            doc_matrix = doc_matrix[:training_no, :, :]
            score_matrix = score_matrix[:training_no, :]

            if index == batches - 1:
                save_model = True

            init_ep = index if index >= 0 else 0

            print(weights)
            print("index: " + str(index))
            print("init_epoch: " + str(init_ep))
            train_model(model, model_name, doc_matrix, score_matrix, init_ep, init_ep + epochs,
                        batch_size=batch_size, val_size=val_size, save_model=save_model)


if __name__ == "__main__":
    name = str(sys.argv[1])
    bs = int(sys.argv[2])
    ep = int(sys.argv[3])
    bn = bool(int(sys.argv[4]))
    dset = str(sys.argv[5])
    if len(sys.argv) > 7:
        w1 = float(sys.argv[6])
        w2 = float(sys.argv[7])
        train(name, bs, ep, bn, dset, (w1, w2))
    else:
        train(name, bs, ep, bn, dset)
