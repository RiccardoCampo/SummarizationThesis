import sys

from dataset import get_matrices
from summarization import build_model, train_model


def train(model_name, batch_size, epochs):
    binary = False
    weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                    (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                    (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    batches = 15
    training_no = 666  # includes validation.
    doc_size = 300
    vector_size = 134

    for weights in weights_list:
        model_name += "_" + str(batch_size) + "_" + str(epochs) + "_" + str(weights)
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


if __name__ == "__main__":
    name = str(sys.argv[1])
    bs = sys.argv[2]
    ep = sys.argv[3]
    train(name, bs, ep)
