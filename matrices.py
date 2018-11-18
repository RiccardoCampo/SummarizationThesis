import sys

from dataset_scores import store_full_sentence_matrices, store_score_matrices


def matrices(ref, batch):
    print("matrices {}".format(batch))
    #store_full_sentence_matrices(batch, ref)
    if ref:
        for scores in ("non_bin", "bin", "bestN"):
            print("scores: {} {}".format(batch, scores))
            store_score_matrices(batch, scores, True)


if __name__ == "__main__":
    ref = bool(int(sys.argv[1]))
    batch = int(sys.argv[2])
    matrices(ref, batch)