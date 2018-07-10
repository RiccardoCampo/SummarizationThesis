from math import sqrt

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans

from pas import extract_pas, realize_pas
from pyrouge import Rouge155


# Return the summary of the source text given the source text, the max length and the weights.
def summarize(source_text, max_length, weights, dataset_name):
    # Extracting PASs and clustering them.
    pas_list = extract_pas(source_text, dataset_name)
    # clustering_pas(pas_list, weights)

    # Creating a list of PASs and relative scores then ordering it.
    sorted_list = []
    for pas in pas_list:
        score = np.array(pas.vector).dot(np.array(weights))
        sorted_list.append((score, realize_pas(pas), pas.position))
    sorted_list.sort(key=lambda tup: tup[0])

    # Selecting best scored PASs until max length is reached
    best_pas_list = []
    summ_len = 0
    for scored_pas in sorted_list:
        if summ_len + len(scored_pas[1]) > max_length:
            break
        best_pas_list.append((scored_pas[2], scored_pas[1]))
        summ_len += len(scored_pas[1])

    sorted_list.sort(key=lambda tup: tup[0], reverse=True)
    summary = ""
    for bp in best_pas_list:
        summary += bp[1]

    return summary


# Return the best PASs of the source text given the source text, the max number of PASs and the weights.
def best_pas(pas_list, max_pas, weights):
    # clustering_pas(pas_list, weights)

    # Creating a list of PASs and relative scores then ordering it.
    sorted_list = []
    for pas in pas_list:
        score = np.array(pas.vector).dot(np.array(weights))
        sorted_list.append((score, pas))
    sorted_list.sort(key=lambda tup: tup[0], reverse=True)

    # Selecting best scored PASs until max length is reached
    best_pas_list = []
    for i in range(min(max_pas, len(pas_list))):
        best_pas_list.append(sorted_list[i][1])

    return best_pas_list


# Retaining the most scored PAS between the couples of PAS which are too similar to each other.
def clustering_pas(pas_list, weights):
    removed_pas = []
    for pas1 in pas_list:
        for pas2 in pas_list:
            # For each couple of different PAS their similarity is computed as the inner product of their embeddings
            if (not pas1 == pas2) and pas1 not in removed_pas and pas2 not in removed_pas:
                if np.inner(np.array(pas1.embeddings), np.array(pas2.embeddings)) > 0.7:
                    print(realize_pas(pas1))
                    print(realize_pas(pas2))
                    if np.array(pas1.vector).dot(np.array(weights)) > np.array(pas2.vector).dot(np.array(weights)):
                        pas_list.remove(pas2)
                        removed_pas.append(pas2)
                    else:
                        pas_list.remove(pas1)
                        removed_pas.append(pas2)


# Return the ROUGE evaluation given source and reference summary
def summary_rouge_score(source_text, reference, max_length, weights, dataset_name):
    # ROUGE package needs to read model(reference) and system(computed) summary from specific folders,
    # so temp files are created to store these two.
    temp_system = open("C:/Users/Riccardo/Desktop/tesi_tmp/rouge/system_summaries/001.txt", "w+")
    temp_model = open("C:/Users/Riccardo/Desktop/tesi_tmp/rouge/model_summaries/001.txt", "w+")

    summary = summarize(source_text, max_length, weights, dataset_name)
    print("SUMMARY" + "\n" + summary)
    print("\n\n\n")
    print("REFERENCE" + "\n" + reference)
    print(summary, file=temp_system)
    print(reference, file=temp_model)

    temp_model.close()
    temp_system.close()

    r = Rouge155()
    r.system_dir = 'C:/Users/Riccardo/Desktop/tesi_tmp/rouge/system_summaries'
    r.model_dir = 'C:/Users/Riccardo/Desktop/tesi_tmp/rouge/model_summaries'
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '(\d+).txt'

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    return output_dict


def summary_clustering_score(document_vectors, summary_vectors, reference_vectors):
    # Removing zero rows from vectors lists.
    document_vectors = document_vectors[~np.all(document_vectors == 0, axis=1)]
    summary_vectors = summary_vectors[~np.all(summary_vectors == 0, axis=1)]
    reference_vectors = reference_vectors[~np.all(reference_vectors == 0, axis=1)]

    vector_dim = len(document_vectors[0])
    centroid = [np.mean(np.array([vec[i] for vec in reference_vectors])) for i in range(len(reference_vectors[0]))]
    init_centroids = np.array([[0] * vector_dim, centroid])
    #init_centroids = np.array([[0.49821944, 0.56260339, 0.17955776, 0.00185936, 0.50955343, 0.12072317],
     #                          [0.69208472, 0.71773158, 0.3029794, 0.00217993, 0.67792622, 1.11437755]])
    X = np.concatenate((document_vectors, reference_vectors))
    kmeans = KMeans(n_clusters=2,
                    #init=init_centroids,
                     ).fit(X)
    print("DOC + REF VECTORS WITH CENTROID")
    print(kmeans.cluster_centers_)
    print(kmeans.predict(summary_vectors))
    print(kmeans.predict(document_vectors))
    print(kmeans.predict(reference_vectors))

    vec_perc = np.count_nonzero(kmeans.predict(document_vectors)) / len(document_vectors)
    print("Original text selected pas percentage:")
    print("{:.3%}".format(vec_perc))

    ref_perc = np.count_nonzero(kmeans.predict(reference_vectors)) / len(reference_vectors)
    print("Reference text selected pas percentage:")
    print("{:.3%}".format(ref_perc))

    best_perc = np.count_nonzero(kmeans.predict(summary_vectors)) / len(summary_vectors)
    print("Best pas selected pas percentage:")
    print("{:.3%}".format(best_perc))

    if vec_perc + best_perc:
        pred_score = best_perc / (vec_perc + best_perc)
    else:
        pred_score = 0
    print("Correct prediction of best pas score:")
    print("{:.3%}".format(pred_score))
    print()

    return vec_perc, ref_perc, best_perc, pred_score


def summary_clustering_score_2(document_vectors, summary_vectors, reference_vectors):
    # Removing zero rows from vectors lists.
    document_vectors = document_vectors[~np.all(document_vectors == 0, axis=1)]
    summary_vectors = summary_vectors[~np.all(summary_vectors == 0, axis=1)]
    reference_vectors = reference_vectors[~np.all(reference_vectors == 0, axis=1)]

    X = np.concatenate((document_vectors, reference_vectors))
    kmeans = KMeans(n_clusters=len(reference_vectors), init=reference_vectors).fit(X)
    print("DOC + REF VECTORS WITH CENTROID")
    print(kmeans.predict(summary_vectors))
    print(kmeans.predict(document_vectors))
    print(kmeans.predict(reference_vectors))

    clusters_no = np.unique(kmeans.predict(reference_vectors)).size
    doc_clusters = kmeans.predict(document_vectors)
    doc_coverage = np.zeros(clusters_no)
    for i in range(clusters_no):
        doc_coverage[i] = (doc_clusters == i).sum()

    print("hello")
    # How many clusters of reference summaries are covered with the generated summary
    summ_coverage = np.unique(kmeans.predict(summary_vectors)).size / clusters_no
    # Adjusted coverage takes into account the fact that the document itslef may not cover the reference summary.
    summ_adj_coverage = np.unique(kmeans.predict(summary_vectors)).size / np.count_nonzero(doc_coverage)

    print("DOCUMENT COVERAGE:")
    print(doc_coverage)
    print()

    print("SUMMARY COVERAGE:")
    print(summ_coverage)

    print("SUMMARY ADJUSTED COVERAGE:")
    print(summ_adj_coverage)

    return summ_coverage, summ_adj_coverage