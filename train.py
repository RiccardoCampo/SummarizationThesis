import sys

from dataset import get_matrices
from summarization import build_model, train_model
from utils import plot_history


def train(series_name, batch_size, epochs, binary, weights=None):
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
            doc_matrix, ref_matrix, score_matrix = get_matrices(weights=weights, binary=binary, index=index)

            if index == batches - 1:
                save_model = True

            print(weights)
            print("index: " + str(index))
            train_model(model, model_name, doc_matrix[:training_no, :, :], score_matrix[:training_no, :], epochs=epochs,
                        batch_size=batch_size, save_model=save_model)
            plot_history(model_name)


if __name__ == "__main__":
    name = str(sys.argv[1])
    bs = int(sys.argv[2])
    ep = int(sys.argv[3])
    bn = bool(sys.argv[4])
    if sys.argv[5]:
        w1 = float(sys.argv[5])
        w2 = float(sys.argv[6])
        train(name, bs, ep, bn, (w1, w2))
    else:
        train(name, bs, ep, bn)
