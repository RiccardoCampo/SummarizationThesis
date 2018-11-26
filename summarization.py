import os
import random

import numpy as np

from pyrouge import Rouge155
from deep_model import predict_scores
from utils import sample_summaries, direct_speech_ratio, get_sources_from_pas_lists, tokens, text_cleanup, \
    resolve_anaphora_pas_list, sentence_embeddings, centrality_scores, tf_idf, stem_and_stopword, \
    direct_speech_ratio_pas


def generate_summary(pas_list, scores, summ_len=100):
    """
    Generate the summary given the scores and the source text in the form of pas list.

    :param pas_list: input pas list to summarize.
    :param scores: vector containing each pas's score.
    :param summ_len: maximum length of the generated summary.
    :return: string containing the summary.
    """
    #resolve_anaphora_pas_list(pas_list)
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


def generate_extract_summary(sentences, scores, summ_len=100):
    """
    Generate the summary given the scores and the source text in the form of sentence list.
    Simply selects the best scoring sentences until the length is reached.

    :param sentences: list of sentences.
    :param scores: vector with sentences' score.
    :param summ_len: maximum length of the generated summary.
    :return: string containing the summary.
    """
    sents_no = len(sentences)

    scores = scores[:sents_no]
    sorted_scores = [(j, scores[j]) for j in range(len(scores))]
    sorted_scores.sort(key=lambda tup: -tup[1])

    # Get the indices of the sorted pas, then a list of the sorted realized pas.
    sorted_indices = [sorted_score[0] for sorted_score in sorted_scores]

    best_indices = []
    size = 0
    j = 0
    while size < summ_len and j < len(scores):
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


def best_pas(pas_list, max_pas, weights):
    """
    Return the best PASs picking pas with the higher weighted sum of the features until max_pas are selected.

    :param pas_list: input pas list.
    :param max_pas: number of selected pas.
    :param weights: list of weights.
    :return: list of best scoring pas.
    """
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


def document_rouge_scores(summary, reference):
    """
    Return the ROUGE evaluation given source and reference summary.

    :param summary: system generated summary.
    :param reference: human generated reference summary.
    :return: rouge scores.
    """
    # ROUGE package needs to read model(reference) and system(computed) summary from specific folders,
    # so temp files are created to store these two.
    system_path = os.getcwd() + "/temp/system_summaries/"
    model_path = os.getcwd() + "/temp/model_summaries/"
    with open(system_path + "0.txt", "w") as temp_system:
        print(summary, file=temp_system)
    with open(model_path + "0.txt", "w") as temp_model:
        print(reference, file=temp_model)

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


def dataset_rouge_scores_deep(model_name, docs_pas_lists, doc_matrix, refs,
                              dynamic_summ_len=False, batch=0, rem_ds=False):
    """
    Compute rouge scores given a model.

    :param model_name: model name.
    :param docs_pas_lists: list of document pas lists.
    :param doc_matrix: document matrix.
    :param refs: list of reference summaries.
    :param dynamic_summ_len: if True summary length will be the same as the reference summary.
    :param batch: number of the batch to process.
    :param rem_ds: if True sentences with more that 15% of direct speech will not be considered.
    :return: average rouge scores and list of rouge 1 recall scores for each document.
    """
    rouge_scores = {"rouge_1_recall": [], "rouge_1_precision": [], "rouge_1_f_score": [], "rouge_2_recall": [],
                    "rouge_2_precision": [], "rouge_2_f_score": []}

    pred_scores = predict_scores(model_name, doc_matrix)
    summaries = []
    # Store the docs and refs which are not discarded,
    # they are needed to match the order of the summaries while using sample_summaries().
    selected_docs = []
    selected_refs = []
    docs = get_sources_from_pas_lists(docs_pas_lists)

    # Computing the score for each document than compute the average.
    for i in range(len(docs_pas_lists)):
        if direct_speech_ratio_pas(docs_pas_lists[i]) < 0.15 or not rem_ds:
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
            score = document_rouge_scores(summary, refs[i])
            rouge_scores["rouge_1_recall"].append(float(score["rouge_1_recall"]))
            rouge_scores["rouge_1_precision"].append(float(score["rouge_1_precision"]))
            rouge_scores["rouge_1_f_score"].append(float(score["rouge_1_f_score"]))
            rouge_scores["rouge_2_recall"].append(float(score["rouge_2_recall"]))
            rouge_scores["rouge_2_precision"].append(float(score["rouge_2_precision"]))
            rouge_scores["rouge_2_f_score"].append(float(score["rouge_2_f_score"]))

    sample_summaries(model_name, selected_docs, selected_refs, summaries, rouge_scores["rouge_1_recall"], batch=batch, all=True)

    return rouge_scores


#
def dataset_rouge_scores_extract(model_name, docs, doc_matrix, refs, dynamic_summ_len=False, batch=0, rem_ds=False):
    """
    Compute rouge scores given a model (Extractive summaries).

    :param model_name: model name.
    :param docs: list of documents.
    :param doc_matrix: document matrix.
    :param refs: list of reference summaries.
    :param dynamic_summ_len: if True summary length will be the same as the reference summary.
    :param batch: number of the batch to process.
    :param rem_ds: if True sentences with more that 15% of direct speech will not be considered.
    :return: average rouge scores and list of rouge 1 recall scores for each document.
    """
    rouge_scores = {"rouge_1_recall": [], "rouge_1_precision": [], "rouge_1_f_score": [], "rouge_2_recall": [],
                    "rouge_2_precision": [], "rouge_2_f_score": []}

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
            score = document_rouge_scores(summary, refs[i])
            rouge_scores["rouge_1_recall"].append(score["rouge_1_recall"])
            rouge_scores["rouge_1_precision"].append(score["rouge_1_precision"])
            rouge_scores["rouge_1_f_score"].append(score["rouge_1_f_score"])
            rouge_scores["rouge_2_recall"].append(score["rouge_2_recall"])
            rouge_scores["rouge_2_precision"].append(score["rouge_2_precision"])
            rouge_scores["rouge_2_f_score"].append(score["rouge_2_f_score"])

        sample_summaries(model_name, selected_docs, selected_refs, summaries, rouge_scores["rouge_1_recall"],
                         batch=batch, all=True)

        return rouge_scores


def dataset_rouge_scores_weighted(docs_pas_lists, refs, weights, ds_threshold=0.15, summ_len=100):
    """
    Computing rouge scores with the weighted method.

    :param docs_pas_lists: list of document pas lists.
    :param refs: list of reference summaries.
    :param weights: list of weights
    :param summ_len: maximum length of the generated summary.
    :return: rouge scores.
    """

    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}
    docs_no = 0
    # Same operations as for the previous function except that the scores are computed by multiplying the features by
    # the weights.
    ds = 0
    for pas_list in docs_pas_lists:
        #weights = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1),
         #          random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        #print(weights)
        #weights = np.random.dirichlet(np.ones(6),size=1)[0]
        if direct_speech_ratio_pas(pas_list) < ds_threshold:
            scores = [np.array(pas.vector).dot(np.array(weights)) for pas in pas_list]
            #scores = np.random.randint(0,1000,len(pas_list))
            summary = generate_summary(pas_list, scores, summ_len=summ_len)

            score = document_rouge_scores(summary, refs[docs_pas_lists.index(pas_list)])
            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            docs_no += 1
        else:
            ds += 1

    print("ds: ")
    print(ds)

    for k in rouge_scores.keys():
        rouge_scores[k] /= docs_no
    return rouge_scores


def dataset_rouge_scores_weighted_extractive(docs_sent_lists, vectors_lists, refs, weights, summ_len=100):
    """
    Computing rouge scores with the weighted method.

    :param docs_pas_lists: list of document pas lists.
    :param refs: list of reference summaries.
    :param weights: list of weights
    :param summ_len: maximum length of the generated summary.
    :return: rouge scores.
    """
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}
    docs_no = 0
    # Same operations as for the previous function except that the scores are computed by multiplying the features by
    # the weights.
    for sentences in docs_sent_lists:
        sent_index = docs_sent_lists.index(sentences)
        if direct_speech_ratio(sentences) < 0.15:
            scores = [vector.dot(np.array(weights)) for vector in vectors_lists[sent_index]]
            #scores = np.random.randint(0, 1000, len(sentences))
            summary = generate_extract_summary(sentences, scores, summ_len=summ_len)

            score = document_rouge_scores(summary, refs[sent_index])
            rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
            rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
            rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
            rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
            rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
            rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]
            docs_no += 1

    for k in rouge_scores.keys():
        rouge_scores[k] /= docs_no
    return rouge_scores
