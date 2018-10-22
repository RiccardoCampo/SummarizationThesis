import os
import pickle

import numpy as np
from keras.callbacks import TensorBoard

from numpy.linalg import norm
from sklearn.cluster import KMeans
from pyrouge import Rouge155

from keras import Input, Model
from keras.engine.saving import load_model
from keras import backend as K
from keras.layers import Dense, LSTM, Bidirectional, Masking, Lambda, Activation


# Return the best PASs of the source text given the source text, the max number of PASs and the weights.
from utils import sample_summaries, direct_speech_ratio, get_sources_from_pas_lists, tokens, text_cleanup


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

    # Amplifying the magnitude.
    # average = np.average(scores[~np.all(scores == 0)])
    # maximum = max(scores)
    # scores = [(((score + average)/(maximum + average)) if score > average else score) for score in scores]

    # Trun scores into 1s and 0s (best 1/3 of the document)
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


# Initialize and compile a model for the specific dimensions.
def build_model(doc_size, vector_size, loss_function, dense_layers, output_activation):
    inputs = Input(shape=(doc_size, vector_size))
    mask = Masking(mask_value=0.0)(inputs)

    blstm = Bidirectional(LSTM(1, return_sequences=True), merge_mode="ave")(mask)
    blstm = Lambda(lambda x: K.squeeze(x, -1))(blstm)

    # If more than one dense layer is present between each dense layer a "relu" is placed to propagate the gradient.
    # If no dense layer is used the specified activation is directly applied.
    if dense_layers > 0:
        blstm_act = Activation("relu")(blstm)
        dense = Dense(doc_size)(blstm_act)
        for i in range(1, dense_layers):
            dense_act = Activation("relu")(dense)
            dense = Dense(doc_size)(dense_act)
        output = Activation(output_activation)(dense)
    else:
        output = Activation(output_activation)(blstm)

    # output = Lambda(crop)([output, inputs])

    model = Model(inputs=inputs, outputs=output)
    model.compile('adam', loss=loss_function, metrics=['accuracy'])

    print(model.summary())

    return model


# Train a pre-compiled model with the provided inputs.
def train_model(model, model_name, doc_matrix, score_matrix, initial_epoch,
                epochs, batch_size=1, val_size=0, save_model=False):
    if val_size > 0:
        set_size = int(doc_matrix.shape[0] - val_size)
    else:
        set_size = int(doc_matrix.shape[0] / 2)  # Half for training, half for validation.

    x_train = doc_matrix[:set_size, :, :]
    x_test = doc_matrix[set_size:, :, :]
    y_train = score_matrix[:set_size, :]
    y_test = score_matrix[set_size:, :]

    log_path = os.getcwd() + "/results/logs/" + model_name
    tensorboard = TensorBoard(log_dir=log_path)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=[x_test, y_test],
                        callbacks=[tensorboard],
                        initial_epoch=initial_epoch)

    history_path = os.getcwd() + "/results/histories/" + model_name + ".hst"
    history = history.history
    if os.path.isfile(history_path):
        with open(history_path, "rb") as file:
            old_history = pickle.load(file)
        for key in history.keys():
            old_history[key].extend(history[key])
        history = old_history

    with open(history_path, "wb") as dest_file:
        pickle.dump(history, dest_file)

    if save_model:
        model.save(os.getcwd() + "/models/" + model_name + ".h5")
        K.clear_session()


# Crops the output(x[0]) based on the input(x[1]) padding.
def crop(x):
    dense = x[0]
    inputs = x[1]
    vector_size = 134

    # Build a matrix having 1 for every non-zero vector, 0 otherwise.
    padding = K.cast(K.not_equal(inputs, 0), dtype=K.floatx())  # Shape: BxDxV.
    # Transposing the matrix.
    padding = K.permute_dimensions(padding, (0, 2, 1))  # Shape: BxVxD.

    resizing = K.ones((1, vector_size))  # Shape: 1xV.
    padding = K.dot(resizing, padding)  # Shape: Bx1xD
    padding = K.squeeze(padding, 0)
    # Rebuilding the vector with only 1 and 0 (as the dot will produce vector_size and 0s).
    padding = K.cast(K.not_equal(padding, 0), dtype=K.floatx())

    # Multiplying the output by the padding (thus putting to zero the padding documents).
    return dense * padding


# Returns the predicted scores given model name and documents.
def predict_scores(model_name, docs):
    model = load_model(os.getcwd() + "/models/" + model_name + ".h5")
    return model.predict(docs, batch_size=1)


# Return the ROUGE evaluation given source and reference summary
def rouge_score(summaries, references):
    # ROUGE package needs to read model(reference) and system(computed) summary from specific folders,
    # so temp files are created to store these two.

    system_path = os.getcwd() + "/temp/system_summaries/"
    model_path = os.getcwd() + "/temp/model_summaries/"

    for i in range(len(summaries)):
        with open(system_path + str(i) + ".txt", "w") as temp_system:
            print(summaries[i], file=temp_system)
        with open(model_path + str(i) + ".txt", "w") as temp_model:
            print(references[i], file=temp_model)

    if os.name == "posix":
        r = Rouge155("/home/arcslab/Documents/Riccardo_Campo/tools/ROUGE-1.5.5")
    else:
        r = Rouge155()
    r.system_dir = system_path
    r.model_dir = model_path
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '(\d+).txt'

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    return output_dict


# Generate the summary give the Model and the source text in the form of pas list.
def generate_summary(pas_list, scores, summ_len=100):
    pas_no = len(pas_list)
    sorted_scores = [(j, scores[j]) for j in range(len(scores))]
    sorted_scores.sort(key=lambda tup: -tup[1])

    # Get the indices of the sorted pas, then a list of the sorted realized pas.
    sorted_indices = [sorted_score[0] for sorted_score in sorted_scores]
    sorted_pas = [pas_list[index] for index in sorted_indices]

    # Build a list of best pas taking at most one pas per sentence.
    best_indices_list = []
    # A dictionary containing a tuple (realized pas, index) for each used sentence).
    pas_per_sentence = {}
    size = 0
    j = 0
    while size < summ_len and j < pas_no:
        current_pas = sorted_pas[j]
        current_index = sorted_indices[j]

        # Excluding short pas (errors).
        if len(current_pas.realized_pas.split()) > 3:
            # If the sentence has never been used it is added to the list.
            # Otherwise between the previous and the current pas the longest one is selected.
            # (To do this it is necessary to decrease the size and remove the previous pas index).
            if current_pas.sentence not in pas_per_sentence.keys():
                best_indices_list.append(current_index)
                pas_per_sentence[current_pas.sentence] = (current_pas.realized_pas, current_index)
                size += len(current_pas.realized_pas.split())
            else:
                if len(current_pas.realized_pas) > len(pas_per_sentence[current_pas.sentence][0]):
                    size -= len(pas_per_sentence[current_pas.sentence][0])
                    best_indices_list.remove(pas_per_sentence[current_pas.sentence][1])
                    best_indices_list.append(current_index)
                    pas_per_sentence[current_pas.sentence] = (current_pas.realized_pas, current_index)
                    size += len(pas_per_sentence[current_pas.sentence][0])
        j += 1

    # Sort the best indices and build the summary.
    best_indices_list.sort()

    summary = ""
    for index in best_indices_list:
        summary += pas_list[index].realized_pas + ".\n"

    return summary


# Generate the summary give the Model and the source text in the form of pas list.
def generate_extract_summary(sentences, scores, summ_len=100):
    sents_no = len(sentences)

    scores = scores[:sents_no]
    sorted_scores = [(j, scores[j]) for j in range(len(scores))]
    sorted_scores.sort(key=lambda tup: -tup[1])

    # Get the indices of the sorted pas, then a list of the sorted realized pas.
    sorted_indices = [sorted_score[0] for sorted_score in sorted_scores]

    best_indices = []
    size = 0
    j = 0
    while size < summ_len and j < sents_no:
        index = sorted_indices[j]
        sent_len = len(sentences[index].split())
        if sent_len > 3:
            if size + sent_len < summ_len:
                best_indices.append(index)
                size += sent_len
        j += 1

    best_indices.sort()

    summary = ""
    for index in best_indices:
        summary += sentences[index] + "\n"

    return summary


# Compute rouge scores given a model.
def testing(model_name, docs_pas_lists, doc_matrix, refs, dynamic_summ_len=False, batch=0, rem_ds=False):
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}
    recall_score_list = []

    pred_scores = predict_scores(model_name, doc_matrix)
    summaries = []
    # Store the docs and refs which are not discarded,
    # they are needed to match the order of the summaries while using sample_summaries().
    selected_docs = []
    selected_refs = []
    docs = get_sources_from_pas_lists(docs_pas_lists)

    # Computing the score for each document than compute the average.
    for i in range(len(docs_pas_lists)):
        if direct_speech_ratio(docs[i]) < 0.15 or not rem_ds:
            print("Processing doc:" + str(i) + "/" + str(len(docs_pas_lists)))
            pas_no = len(docs_pas_lists[i])

            # Cutting the scores to the length of the document and arrange them by score
            # preserving the original position.
            scores = pred_scores[i][:pas_no]
            if dynamic_summ_len:
                summary = generate_summary(docs_pas_lists[i], scores, summ_len=len(refs[i].split()))
            else:
                summary = generate_summary(docs_pas_lists[i], scores, summ_len=100)
            summaries.append(summary)
            selected_docs.append(docs[i])
            selected_refs.append(refs[i])

            # Get the rouge scores.
            score = rouge_score([summary], [refs[i]])
            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            recall_score_list.append(score["rouge_1_recall"])

    sample_summaries(model_name, selected_docs, selected_refs, summaries, recall_score_list, batch=batch)

    for k in rouge_scores.keys():
        rouge_scores[k] /= len(summaries)

    return rouge_scores, recall_score_list


# Compute rouge scores given a model.
def testing_extract(model_name, docs, doc_matrix, refs, dynamic_summ_len=False, batch=0, rem_ds=False):
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}
    recall_score_list = []

    pred_scores = predict_scores(model_name, doc_matrix)
    summaries = []
    # Store the docs and refs which are not discarded,
    # they are needed to match the order of the summaries while using sample_summaries().
    selected_docs = []
    selected_refs = []

    # Computing the score for each document than compute the average.
    for i in range(len(docs)):
        if direct_speech_ratio(docs[i]) < 0.15 or not rem_ds:
            print("Processing doc:" + str(i) + "/" + str(len(docs)))
            docs[i] = text_cleanup(docs[i])
            refs[i] = text_cleanup(refs[i])
            sentences = tokens(docs[i])
            sents_no = len(sentences)
            # Cutting the scores to the length of the document and arrange them by score
            # preserving the original position.
            scores = pred_scores[i][:sents_no]
            if dynamic_summ_len:
                summary = generate_extract_summary(sentences, scores, summ_len=len(refs[i].split()))
            else:
                summary = generate_extract_summary(sentences, scores, summ_len=100)
            summaries.append(summary)
            selected_docs.append(docs[i])
            selected_refs.append(refs[i])

            # Get the rouge scores.
            score = rouge_score([summary], [refs[i]])
            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            recall_score_list.append(score["rouge_1_recall"])

    sample_summaries(model_name, selected_docs, selected_refs, summaries, recall_score_list, batch=batch)

    for k in rouge_scores.keys():
        rouge_scores[k] /= len(summaries)

    return rouge_scores, recall_score_list


# Getting the scores with the weighted method.
def testing_weighted(docs_pas_lists, refs, weights, summ_len=100):
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}

    # Same operations as for the previous function except that the scores are computed by multiplying the features by
    # the weights.
    for pas_list in docs_pas_lists:
        scores = [np.array(pas.vector).dot(np.array(weights)) for pas in pas_list]
        summary = generate_summary(pas_list, scores, summ_len=summ_len)

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

