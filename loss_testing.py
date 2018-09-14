from math import sqrt
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans


def summary_clustering_score(document_vectors, summary_vectors, reference_vectors, manage_outliers=True, log=True):
    # Removing zero rows from vectors lists.
    document_vectors = document_vectors[~np.all(document_vectors == 0, axis=1)]
    summary_vectors = summary_vectors[~np.all(summary_vectors == 0, axis=1)]
    reference_vectors = reference_vectors[~np.all(reference_vectors == 0, axis=1)]

    vector_dim = len(document_vectors[0])
    centroid = [np.mean(np.array([vec[i] for vec in reference_vectors])) for i in range(len(reference_vectors[0]))]
  #  init_centroids = np.array([[0] * vector_dim, centroid])

 #   init_centroids = np.array(new_points([0.49821944, 0.56260339, 0.17955776, 0.00185936, 0.50955343, 0.12072317],
  #                             [0.69208472, 0.71773158, 0.3029794, 0.00217993, 0.67792622, 1.11437755], 1, 6))

    init_centroids = np.array(new_points([0] * vector_dim, centroid, 1, 6))


    X = np.concatenate((document_vectors, reference_vectors))
    kmeans = KMeans(n_clusters=2, init=init_centroids).fit(X)

    clusters = kmeans.cluster_centers_
    doc_predict = kmeans.predict(document_vectors)
    ref_predict = kmeans.predict(reference_vectors)
    best_predict = kmeans.predict(summary_vectors)
    selected_doc_no = np.count_nonzero(doc_predict)
    selected_ref_no = np.count_nonzero(ref_predict)

    if selected_doc_no < selected_ref_no and manage_outliers:
        distances = [(i, norm(document_vectors[i] - clusters[1])) for i in range(len(document_vectors))]
        distances.sort(key=lambda tup: tup[1])
        for i in range(min(selected_ref_no, len(document_vectors))):
            index = distances[i][0]
            doc_predict[index] = 1
            for j in range(len(best_predict)):
                if (document_vectors[index] == summary_vectors[j]).all():
                    best_predict[j] = 1

    vec_perc = np.count_nonzero(doc_predict) / len(document_vectors)
    ref_perc = np.count_nonzero(ref_predict) / len(reference_vectors)
    best_perc = np.count_nonzero(best_predict) / len(summary_vectors)

    if vec_perc + best_perc:
        pred_score = best_perc / (vec_perc + best_perc)
    else:
        pred_score = 0

    len_diff = (len(document_vectors) - len(summary_vectors)) / len(document_vectors)
    if len_diff:
        adjusted_score = best_perc - vec_perc / len_diff
    else:
        adjusted_score = pred_score
    if not adjusted_score:
        adjusted_score = 0

    if log:
        print(init_centroids)
        print("DOC + REF VECTORS WITH CENTROID")
        print(kmeans.cluster_centers_)
        print(best_predict)
        print(doc_predict)
        print(ref_predict)

        print("Original text selected pas percentage:")
        print("{:.3%}".format(vec_perc))


        print("Reference text selected pas percentage:")
        print("{:.3%}".format(ref_perc))


        print("Best pas selected pas percentage:")
        print("{:.3%}".format(best_perc))

        print("Correct prediction of best pas score:")
        print("{:.3%}".format(pred_score))

        print("Correct prediction of best pas adjusted score:")
        print("{:.3%}".format(pred_score))
        print()

    return vec_perc, ref_perc, best_perc, pred_score, adjusted_score


def summary_clustering_score_2(document_vectors, summary_vectors, reference_vectors, log=True):
    # Removing zero rows from vectors lists.
    document_vectors = document_vectors[~np.all(document_vectors == 0, axis=1)]
    summary_vectors = summary_vectors[~np.all(summary_vectors == 0, axis=1)]
    reference_vectors = reference_vectors[~np.all(reference_vectors == 0, axis=1)]

    X = np.concatenate((document_vectors, reference_vectors))
    kmeans = KMeans(n_clusters=len(reference_vectors), init=reference_vectors).fit(X)

    clusters_no = np.unique(kmeans.predict(reference_vectors)).size
    doc_clusters = kmeans.predict(document_vectors)
    doc_coverage = np.zeros(clusters_no)
    for i in range(clusters_no):
        doc_coverage[i] = (doc_clusters == i).sum()

    # How many clusters of reference summaries are covered with the generated summary
    summ_coverage = np.unique(kmeans.predict(summary_vectors)).size / clusters_no
    # Adjusted coverage takes into account the fact that the document itslef may not cover the reference summary.
    summ_adj_coverage = np.unique(kmeans.predict(summary_vectors)).size / np.count_nonzero(doc_coverage)

    if log:
        print("DOC + REF VECTORS WITH CENTROID")
        print(kmeans.predict(summary_vectors))
        print(kmeans.predict(document_vectors))
        print(kmeans.predict(reference_vectors))

        print("DOCUMENT COVERAGE:")
        print(doc_coverage)
        print()

        print("SUMMARY COVERAGE:")
        print(summ_coverage)

        print("SUMMARY ADJUSTED COVERAGE:")
        print(summ_adj_coverage)

    return summ_coverage, summ_adj_coverage


def summary_clustering_score_3(document_vectors, summary_vectors, reference_vectors, log=True):
    # Removing zero rows from vectors lists.
    document_vectors = document_vectors[~np.all(document_vectors == 0, axis=1)]
    summary_vectors = summary_vectors[~np.all(summary_vectors == 0, axis=1)]
    reference_vectors = reference_vectors[~np.all(reference_vectors == 0, axis=1)]
    vector_dim = len(document_vectors[0])

    factor = sqrt(128)
    for i in range(len(document_vectors)):
        for j in range(6, vector_dim):
            document_vectors[i][j] = document_vectors[i][j] / factor
            if i < len(summary_vectors):
                summary_vectors[i][j] = summary_vectors[i][j] / factor
                reference_vectors[i][j] = reference_vectors[i][j] / factor

    centroid = [np.mean(np.array([vec[i] for vec in reference_vectors])) for i in range(len(reference_vectors[0]))]
    #init_centroids = np.array([[-1] * vector_dim, centroid])

    #   init_centroids = np.array(new_points([0.49821944, 0.56260339, 0.17955776, 0.00185936, 0.50955343, 0.12072317],
    #                             [0.69208472, 0.71773158, 0.3029794, 0.00217993, 0.67792622, 1.11437755], 1, 6))

    init_centroids = np.array(new_points([0] * vector_dim, centroid, 1, 134))

    print(init_centroids)

    X = np.concatenate((document_vectors, reference_vectors))
    kmeans = KMeans(n_clusters=2,
                    init=init_centroids,
                    ).fit(X)
    vec_perc = np.count_nonzero(kmeans.predict(document_vectors)) / len(document_vectors)
    ref_perc = np.count_nonzero(kmeans.predict(reference_vectors)) / len(reference_vectors)
    best_perc = np.count_nonzero(kmeans.predict(summary_vectors)) / len(summary_vectors)
    if vec_perc + best_perc:
        pred_score = best_perc / (vec_perc + best_perc)
    else:
        pred_score = 0

    adjusted_score = best_perc - vec_perc / (len(document_vectors) - len(summary_vectors)) * len(document_vectors)
    if not adjusted_score:
        adjusted_score = 0

    if log:
        print("DOC + REF VECTORS WITH CENTROID")
        print(kmeans.cluster_centers_)
        print(kmeans.predict(summary_vectors))
        print(kmeans.predict(document_vectors))
        print(kmeans.predict(reference_vectors))

        print("Original text selected pas percentage:")
        print("{:.3%}".format(vec_perc))

        print("Reference text selected pas percentage:")
        print("{:.3%}".format(ref_perc))

        print("Best pas selected pas percentage:")
        print("{:.3%}".format(best_perc))

        print("Correct prediction of best pas score:")
        print("{:.3%}".format(pred_score))
        print()

        print("Correct prediction of best pas adjusted score:")
        print("{:.3%}".format(pred_score))
        print()

    return vec_perc, ref_perc, best_perc, pred_score, adjusted_score


def new_points(p_a, p_b, dist, dim):
    p_c = []
    p_d = []
    for i in range(dim):
        p_c.append((p_b[i] - p_a[i]) / (p_b[0] - p_a[0]) * (p_a[0]-dist))
        p_d.append((p_b[i] - p_a[i]) / (p_b[0] - p_a[0]) * (p_b[0] + dist))

    return p_c, p_d
