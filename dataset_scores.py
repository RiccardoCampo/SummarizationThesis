import os
import pickle

import numpy as np


# Assign scores to each pas in the document
from numpy.linalg import norm
from sklearn.cluster import KMeans

from dataset_text import get_pas_lists, get_duc
from utils import text_cleanup, tokens, sentence_embeddings, centrality_scores, tf_idf, stem_and_stopword


def score_document(doc_vectors, ref_vectors, weights, binary):
    max_len = len(doc_vectors)
    scores = np.zeros(max_len)
    doc_vectors = doc_vectors[~np.all(doc_vectors == 0, axis=1)]
    ref_vectors = ref_vectors[~np.all(ref_vectors == 0, axis=1)]
    features_no = 6

    # PART ONE: 0/1 CLUSTERING.
    # Removing zero rows from vectors lists.
    doc_ft_vectors = [doc_vector[:features_no] for doc_vector in doc_vectors]
    ref_ft_vectors = [ref_vector[:features_no] for ref_vector in ref_vectors]

    centroid = [np.mean(np.array([vec[i] for vec in ref_ft_vectors])) for i in range(len(ref_ft_vectors[0]))]
    init_centroids = np.array([[0] * features_no, centroid])

    x = np.concatenate((doc_ft_vectors, ref_ft_vectors))
    kmeans = KMeans(n_clusters=2, init=init_centroids).fit(x)

    predicted = kmeans.predict(doc_ft_vectors)
    for i in range(len(predicted)):
        scores[i] = predicted[i] * weights[0]

    # PART TWO: ONE CLUSTER FOR EACH REF. VECTOR
    doc_emb_vectors = [doc_vector[features_no:] for doc_vector in doc_vectors]
    ref_emb_vectors = [ref_vector[features_no:] for ref_vector in ref_vectors]

    x = np.concatenate((doc_emb_vectors, ref_emb_vectors))
    kmeans = KMeans(n_clusters=len(ref_emb_vectors), init=np.array(ref_emb_vectors)).fit(x)
    centers = kmeans.cluster_centers_

    each = 0
    if each:
        for i in range(len(doc_emb_vectors)):
            distances = [norm(doc_emb_vectors[i] - center) for center in centers]
            scores[i] += (1 - (min(distances) / max(distances))) * weights[1]
    else:
        min_distances = [min([norm(doc_emb_vector - center) for center in centers])
                         for doc_emb_vector in doc_emb_vectors]
        for i in range(len(doc_emb_vectors)):
            score = (1 - (min_distances[i] / max(min_distances))) * weights[1]
            # reserving zero as a padding value.
            scores[i] += score if score > 0 else 0.01

    # Squeezing the values between 0.5 and 1 (to better refer to the sigmoid).
    scores = [(score + (1 - score) * 0.5) for score in scores]

    # Turn scores into 1s and 0s (best 1/3 of the document)
    if binary:
        sorted_indexes = [(i, scores[i]) for i in range(len(scores))]
        sorted_indexes.sort(key=lambda tup: -tup[1])

        scores = np.zeros(max_len)
        # 33% or the number of ref vectors
        for i in range(max(int(len(doc_vectors) / 3), len(ref_vectors))):
            scores[sorted_indexes[i][0]] = 1

    return scores


# Assign scores to each pas in the document
def score_document_bestn(doc_vectors, ref_vectors):
    max_len = len(doc_vectors)
    scores = np.zeros(max_len)
    doc_vectors = doc_vectors[~np.all(doc_vectors == 0, axis=1)]
    ref_vectors = ref_vectors[~np.all(ref_vectors == 0, axis=1)]
    doc_len = len(doc_vectors)
    features_no = 6
    doc_emb_vectors = [doc_vector[features_no:] for doc_vector in doc_vectors]
    ref_emb_vectors = [ref_vector[features_no:] for ref_vector in ref_vectors]

    for ref_emb in ref_emb_vectors:
        distances = [np.linalg.norm(ref_emb - doc_emb) for doc_emb in doc_emb_vectors]
        min_index = distances.index(min(distances))
        while scores[min_index] == 1 and np.any(scores[:doc_len] != np.ones(doc_len)):
            distances[min_index] += 1000
            min_index = distances.index(min(distances))
        scores[min_index] = 1
    return scores


# Matrix representation is computed and stored.
def store_matrices(index):
    if index < 0:
        docs_pas_lists, refs_pas_lists = get_pas_lists(-1)
        dataset_path = "/dataset/duc/duc"
    else:
        docs_pas_lists, refs_pas_lists = get_pas_lists(index)
        dataset_path = "/dataset/nyt/" + str(index) + "/nyt" + str(index)

    # Storing the matrices in the appropriate file, depending on the scoring system.
    doc_path = dataset_path + "_doc_matrix.dat"
    ref_path = dataset_path + "_ref_matrix.dat"

    docs_no = len(docs_pas_lists)                                   # First dimension, documents number.
    # Second dimension, max document length (sparse), fixed in case of nyt.
    max_sent_no = max([len(doc) for doc in docs_pas_lists]) if index < 0 else 300
    # Third dimension, vector representation dimension.
    sent_vec_len = len(docs_pas_lists[0][0].vector) + len(docs_pas_lists[0][0].embeddings)

    # The matrix are initialized as zeros, then they'll filled in with vectors for each docs' sentence.
    refs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))
    docs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))

    for i in range(docs_no):
        for j in range(max_sent_no):
            if j < len(docs_pas_lists[i]):
                docs_3d_matrix[i, j, :] = np.append(docs_pas_lists[i][j].vector, docs_pas_lists[i][j].embeddings)
            if j < len(refs_pas_lists[i]):
                refs_3d_matrix[i, j, :] = np.append(refs_pas_lists[i][j].vector, refs_pas_lists[i][j].embeddings)

    with open(os.getcwd() + ref_path, "wb") as dest_f:
        pickle.dump(refs_3d_matrix, dest_f)
    with open(os.getcwd() + doc_path, "wb") as dest_f:
        pickle.dump(docs_3d_matrix, dest_f)


# Scores are computed and stored.
# score type can be "non_bin" "bin" "bestN"
def store_score_matrices(index, scores_type, extractive):
    if index < 0:
        dataset_path = "/dataset/duc/duc"
    else:
        dataset_path = "/dataset/nyt/" + str(index) + "/nyt" + str(index)

    # Storing the scores in the appropriate file, depending on the scoring system.
    doc_path = dataset_path + "_doc_matrix.dat"
    ref_path = dataset_path + "_ref_matrix.dat"

    with open(os.getcwd() + ref_path, "rb") as dest_f:
        refs_3d_matrix = pickle.load(dest_f)
    with open(os.getcwd() + doc_path, "rb") as dest_f:
        docs_3d_matrix = pickle.load(dest_f)

    docs_no = docs_3d_matrix.shape[0]
    max_sent_no = docs_3d_matrix.shape[1]

    if extractive:
        common_path = dataset_path + "_sent_score_matrix"
    else:
        common_path = dataset_path + "_score_matrix"

    if scores_type == "bestN":
        scores_path = common_path + "_bestn.dat"

        scores_matrix = np.zeros((docs_no, max_sent_no))
        for i in range(docs_no):
            scores_matrix[i] = score_document_bestn(docs_3d_matrix[i, :, :], refs_3d_matrix[i, :, :])
        with open(os.getcwd() + scores_path, "wb") as dest_f:
            pickle.dump(scores_matrix, dest_f)
    else:
        weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                        (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]

        for weights in weights_list:
            if scores_type == "bin":
                scores_path = common_path + str(weights[0]) + "-" + str(weights[1]) + "binary.dat"
            else:
                scores_path = common_path + str(weights[0]) + "-" + str(weights[1]) + ".dat"

            # Build the score matrix document by document.
            scores_matrix = np.zeros((docs_no, max_sent_no))
            for i in range(docs_no):
                scores_matrix[i] = score_document(docs_3d_matrix[i, :, :], refs_3d_matrix[i, :, :],
                                                  weights, scores_type == "bin")
            with open(os.getcwd() + scores_path, "wb") as dest_f:
                pickle.dump(scores_matrix, dest_f)


# Getting the matrices of documents and reference summaries.
def get_matrices(index, scores_type, extractive, weights):
    # Selecting the right path depending on the batch or binary scoring.
    if index < 0:
        dataset_path = "/dataset/duc/duc"
    else:
        dataset_path = "/dataset/nyt/" + str(index) + "/nyt" + str(index)

    if extractive:
        scores_path = dataset_path + "_sent_score_matrix"
        doc_path = dataset_path + "_doc_sent_matrix.dat"
        ref_path = dataset_path + "_ref_sent_matrix.dat"
    else:
        scores_path = dataset_path + "_score_matrix"
        doc_path = dataset_path + "_doc_matrix.dat"
        ref_path = dataset_path + "_ref_matrix.dat"

    if scores_type == "bin":
        scores_path += str(weights[0]) + "-" + str(weights[1]) + "binary.dat"
    elif scores_type == "bestN":
        scores_path += "_bestn.dat"
    else:
        scores_path += str(weights[0]) + "-" + str(weights[1]) + ".dat"

    with open(os.getcwd() + doc_path, "rb") as docs_f:
        doc_matrix = pickle.load(docs_f)
    with open(os.getcwd() + ref_path, "rb") as refs_f:
        ref_matrix = pickle.load(refs_f)
    with open(os.getcwd() + scores_path, "rb") as scores_f:
        score_matrix = pickle.load(scores_f)

    return doc_matrix, ref_matrix, score_matrix


def store_full_sentence_matrices():
    docs, references, _ = get_duc()

    docs_no = len(docs)                                   # First dimension, documents number.
    # Second dimension, max document length (sparse), fixed in case of nyt.
    max_sent_no = 200
    # Third dimension, vector representation dimension.
    sent_vec_len = 134

    # The matrix are initialized as zeros, then they'll filled in with vectors for each docs' sentence.
    refs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))
    docs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))

    # For each document the pas_list is extracted after cleaning the text and tokenizing it.
    for k in range(2):
        if k == 0:
            doc_list = docs
        else:
            doc_list = references

        for i in range(len(doc_list)):
            doc = doc_list[i]
            print("Processing doc " + str(i) + "/" + str(len(docs)))
            doc = text_cleanup(doc)
            # Splitting sentences (by dot).
            sentences = tokens(doc)
            embeddings = sentence_embeddings(sentences)
            centr_scores = centrality_scores(embeddings)
            tf_idfs = tf_idf(sentences, os.getcwd() + "/dataset/duc/duc_idfs.dat")
            # Position score, reference sentence length score, tf_idf, numerical data, centrality, title.
            for j in range(len(sentences)):
                sent = sentences[j]

                position_score = (len(sentences) - j) / len(sentences)
                length_score = len(sent) / max(len(snt) for snt in sentences)
                tf_idf_score = 0
                numerical_score = 0
                centrality_score = centr_scores[j]
                title_sim_score = np.inner(np.array(embeddings[j]), np.array(embeddings[-1]))

                # Computing centrality and tf_idf score.
                terms = list(set(stem_and_stopword(sent)))
                for term in terms:
                    # Due to errors terms may be not present in the tf_idf dictionary.
                    if term in tf_idfs.keys():
                        tf_idf_score += tf_idfs[term]
                    else:
                        tf_idf_score += 0

                    if term.isdigit():
                        numerical_score += 1

                # Some errors in the preprocessing may lead to zero terms, so it is necessary to avoid division by zero.
                if len(terms):
                    tf_idf_score /= len(terms)
                else:
                    tf_idf_score = 0

                if k == 0:
                    docs_3d_matrix[i, j, :] = np.append([position_score, length_score,
                                                         tf_idf_score, numerical_score,
                                                         centrality_score, title_sim_score], embeddings[j])
                else:
                    refs_3d_matrix[i, j, :] = np.append([position_score, length_score,
                                                         tf_idf_score, numerical_score,
                                                         centrality_score, title_sim_score], embeddings[j])

    # Storing the matrices in the appropriate file, depending on the scoring system.
    doc_path = "/dataset/duc/duc_doc_sent_matrix.dat"
    ref_path = "/dataset/duc/duc_ref_sent_matrix.dat"

    with open(os.getcwd() + ref_path, "wb") as dest_f:
        pickle.dump(refs_3d_matrix, dest_f)
    with open(os.getcwd() + doc_path, "wb") as dest_f:
        pickle.dump(docs_3d_matrix, dest_f)
