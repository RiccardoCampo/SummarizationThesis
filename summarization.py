import os
import shutil
import numpy as np

from keras.engine.saving import load_model
from numpy.linalg import norm
from sklearn.cluster import KMeans
from pyrouge import Rouge155

from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Masking


# Return the best PASs of the source text given the source text, the max number of PASs and the weights.
def best_pas(pas_list, max_pas, weights):
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


# Assign scores to each pas in the document
def score_document(doc_vectors, ref_vectors, weights):
    scores = np.zeros(len(doc_vectors))
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
            scores[i] += (1 - (min_distances[i] / max(min_distances))) * weights[1]

    return scores


def training(doc_matrix, score_matrix, model_name):
    set_size = int(doc_matrix.shape[0] / 2)
    doc_size = int(doc_matrix.shape[1])
    vector_size = int(doc_matrix.shape[2])

    print('Loading data...')

    x_train = doc_matrix[:set_size, :, :]
    x_test = doc_matrix[set_size:, :, :]

    y_train = score_matrix[:set_size, :]
    y_test = score_matrix[set_size:, :]

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(doc_size, vector_size)))
    model.add(Bidirectional(LSTM(doc_size, input_shape=(doc_size, vector_size))))
    model.add(Dense(doc_size, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'mse', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=1,
              epochs=4,
              validation_data=[x_test, y_test])

    print(model.predict(doc_matrix[0:1, :, :]))
    print(score_matrix[0])

    model.save(os.getcwd() + "/models/" + model_name + ".h5")


# Returns the predicted scores given model name and documents.
def predict_scores(model_name, docs):
    model = load_model(os.getcwd() + "/models/" + model_name + ".h5")
    return model.predict(docs)


# Return the ROUGE evaluation given source and reference summary
def rouge_score(summaries, references):
    # ROUGE package needs to read model(reference) and system(computed) summary from specific folders,
    # so temp files are created to store these two.

    system_path = "C:/Users/Riccardo/Desktop/tesi_tmp/rouge/system_summaries/"
    model_path = "C:/Users/Riccardo/Desktop/tesi_tmp/rouge/model_summaries/"
    #if os.path.isdir(system_path):
    #    shutil.rmtree(system_path)
    #if os.path.isdir(model_path):
    #    shutil.rmtree(model_path)
    #os.makedirs(system_path)
    #os.makedirs(model_path)

    for i in range(len(summaries)):
        with open(system_path + str(i) + ".txt", "w") as temp_system:
            print(summaries[i], file=temp_system)
        with open(model_path + str(i) + ".txt", "w") as temp_model:
            print(references[i], file=temp_model)

    r = Rouge155()
    r.system_dir = 'C:/Users/Riccardo/Desktop/tesi_tmp/rouge/system_summaries'
    r.model_dir = 'C:/Users/Riccardo/Desktop/tesi_tmp/rouge/model_summaries'
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '(\d+).txt'

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    return output_dict


def testing(model_name, docs_pas_lists, doc_matrix, score_matrix, refs, summ_len=100):
    max_sent_no = doc_matrix.shape[1]

    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}
    pred_matrix = np.zeros((len(docs_pas_lists), max_sent_no))

    model = load_model(os.getcwd() + "/models/" + model_name + ".h5")

    for i in range(len(docs_pas_lists)):
        print("Processing doc:" + str(i) + "/" + str(len(docs_pas_lists)))
        pas_list = docs_pas_lists[i]
        pas_no = len(pas_list)
        sent_vec_len = len(pas_list[0].vector) + len(pas_list[0].embeddings)
        doc_vectors = np.zeros((1, max_sent_no, sent_vec_len))

        for j in range(max_sent_no):
            if j < pas_no:
                doc_vectors[0, j, :] = np.append(pas_list[j].vector, pas_list[j].embeddings)

        pred_scores = model.predict(doc_vectors)[0]
        pred_matrix[i, :] = pred_scores
        scores = pred_scores[:pas_no]
        sorted_scores = [(j, scores[j]) for j in range(len(scores))]
        sorted_scores.sort(key=lambda tup: -tup[1])

        sorted_indices = [sorted_score[0] for sorted_score in sorted_scores]
        sorted_realized_pas = [pas_list[index].realized_pas for index in sorted_indices]
        best_pas_list = []
        best_indices_list = []
        size = 0
        j = 0
        while size < summ_len and j < pas_no:
            redundant_pas = find_redundant_pas(best_pas_list, sorted_realized_pas[j])
            if redundant_pas is None:
                size += len(sorted_realized_pas[j].split())
                if size < summ_len:
                    best_pas_list.append(sorted_realized_pas[j])
                    best_indices_list.append(sorted_indices[j])
            else:
                if redundant_pas in best_pas_list:
                    if size - len(redundant_pas) + len(sorted_realized_pas[j]) < summ_len:
                        size = size - len(redundant_pas) + len(sorted_realized_pas[j])
                        best_pas_list[best_pas_list.index(redundant_pas)] = sorted_realized_pas[j]
                        best_indices_list[best_pas_list.index(redundant_pas)] = sorted_indices[j]
            j += 1

        best_indices_list.sort()

        summary = ""
        for index in best_indices_list:
            summary += pas_list[index].realized_pas + ".\n"

        score = rouge_score([summary], [refs[i]])
        rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
        rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
        rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
        rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
        rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
        rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

    for k in rouge_scores.keys():
        rouge_scores[k] /= len(docs_pas_lists)

    return rouge_scores


def testing_weighted(docs_pas_lists, refs, weights, summ_len=100):
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}

    for pas_list in docs_pas_lists:
        pas_no = len(pas_list)
        scores = [np.array(pas.vector).dot(np.array(weights)) for pas in pas_list]
        sorted_scores = [(j, scores[j]) for j in range(len(scores))]
        sorted_scores.sort(key=lambda tup: -tup[1])

        sorted_indices = [sorted_score[0] for sorted_score in sorted_scores]
        sorted_realized_pas = [pas_list[index].realized_pas for index in sorted_indices]
        best_pas_list = []
        best_indices_list = []
        size = 0
        j = 0
        while size < summ_len and j < pas_no:
            redundant_pas = find_redundant_pas(best_pas_list, sorted_realized_pas[j])
            if redundant_pas is None:
                size += len(sorted_realized_pas[j].split())
                if size < summ_len:
                    best_pas_list.append(sorted_realized_pas[j])
                    best_indices_list.append(sorted_indices[j])
            else:
                if redundant_pas in best_pas_list:
                    if size - len(redundant_pas) + len(sorted_realized_pas[j]) < summ_len:
                        size = size - len(redundant_pas) + len(sorted_realized_pas[j])
                        best_pas_list[best_pas_list.index(redundant_pas)] = sorted_realized_pas[j]
                        best_indices_list[best_pas_list.index(redundant_pas)] = sorted_indices[j]
            j += 1

        best_indices_list.sort()

        summary = ""
        for index in best_indices_list:
            summary += pas_list[index].realized_pas + ".\n"

        score = rouge_score([summary], [refs[docs_pas_lists.index(pas_list)]])
        rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
        rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
        rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
        rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
        rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
        rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

    for k in rouge_scores.keys():
        rouge_scores[k] /= len(docs_pas_lists)
    return rouge_scores


def find_redundant_pas(realized_pas_list, realized_pas):
    redundant_pas = None
    for pas in realized_pas_list:
        if (not pas.find(realized_pas) == -1) or (not pas.find(realized_pas) == -1):
            if len(pas) < len(realized_pas):
                redundant_pas = pas
            else:
                redundant_pas = realized_pas
    return redundant_pas
