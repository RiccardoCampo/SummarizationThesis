import os
import numpy as np
from pyrouge import Rouge155


# Return the best PASs of the source text given the source text, the max number of PASs and the weights.
from deep_model import predict_scores
from utils import sample_summaries, direct_speech_ratio, get_sources_from_pas_lists, tokens, text_cleanup, \
    resolve_anaphora_pas_list


# Generate the summary give the Model and the source text in the form of pas list.
def generate_summary(pas_list, scores, summ_len=100):
    resolve_anaphora_pas_list(pas_list)
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


# Return the ROUGE evaluation given source and reference summary
def document_rouge_scores(summary, reference):
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


# Compute rouge scores given a model.
def dataset_rouge_scores_deep(model_name, docs_pas_lists, doc_matrix, refs,
                              dynamic_summ_len=False, batch=0, rem_ds=False):
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
            score = document_rouge_scores(summary, refs[i])
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
def dataset_rouge_scores_extract(model_name, docs, doc_matrix, refs, dynamic_summ_len=False, batch=0, rem_ds=False):
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
            score = document_rouge_scores(summary, refs[i])
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
def dataset_rouge_scores_weighted(docs_pas_lists, refs, weights, summ_len=100):
    rouge_scores = {"rouge_1_recall": 0, "rouge_1_precision": 0, "rouge_1_f_score": 0, "rouge_2_recall": 0,
                    "rouge_2_precision": 0, "rouge_2_f_score": 0}

    # Same operations as for the previous function except that the scores are computed by multiplying the features by
    # the weights.
    for pas_list in docs_pas_lists:
        scores = [np.array(pas.vector).dot(np.array(weights)) for pas in pas_list]
        summary = generate_summary(pas_list, scores, summ_len=summ_len)

        score = document_rouge_scores(summary, refs[docs_pas_lists.index(pas_list)])
        rouge_scores["rouge_1_recall"] += score["rouge_1_recall"]
        rouge_scores["rouge_1_precision"] += score["rouge_1_precision"]
        rouge_scores["rouge_1_f_score"] += score["rouge_1_f_score"]
        rouge_scores["rouge_2_recall"] += score["rouge_2_recall"]
        rouge_scores["rouge_2_precision"] += score["rouge_2_precision"]
        rouge_scores["rouge_2_f_score"] += score["rouge_2_f_score"]

    for k in rouge_scores.keys():
        rouge_scores[k] /= len(docs_pas_lists)
    return rouge_scores
