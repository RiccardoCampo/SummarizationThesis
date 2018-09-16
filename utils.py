import os
import random
import re
import string
import unicodedata
import time
import pickle

import nltk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import matplotlib

from summarization import generate_summary

matplotlib.use("Pdf")
import matplotlib.pyplot as plt

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
embedder = hub.Module("https://tfhub.dev/google/random-nnlm-en-dim128/1")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

if os.name == "posix":
    result_path = "/home/arcslab/Documents/Riccardo_Campo/results/"
else:
    result_path = "C:/Users/Riccardo/Desktop/temp_results/"


# Preprocessing the input text to make it ready for the SRL
def text_cleanup(full_text):
    # Eliminating new lines as it causes problems with the unicode normalization.
    full_text = full_text.replace("\n", " ")

    # Preserving dash before unicode cleanup.
    full_text = full_text.replace('—', "-")
    full_text = full_text.replace('–', "-")

    # Uniforming apices style
    full_text = full_text.replace("``", '"')
    full_text = full_text.replace("''", '"')

    # Spacing slashes.
    full_text = full_text.replace("/", " / ")

    # Removing dashes like th-is and parentheses like this(these) or (this)these,
    # while preserving cases like - this one - and (this one).
    full_text = re.sub("([a-zA-Z0-9])-([a-zA-Z0-9])", r'\1 \2', full_text)
    full_text = re.sub("([a-zA-Z0-9])\(([a-zA-Z0-9]+)\)", r'\1\2', full_text)
    full_text = re.sub("\(([a-zA-Z0-9]+)\)([a-zA-Z0-9])", r'\1\2', full_text)

    # Spacing the contracted form (I'm to I 'm) to fit the srl format.
    full_text = re.sub("([a-zA-Z0-9])\'([a-zA-Z0-9])", r"\1 '\2", full_text)

    # Spacing apices and parentheses to prevent errors.
    full_text = re.sub("\"([a-zA-Z0-9 ,']+)\"", r'" \1 "', full_text)
    full_text = re.sub("\(([a-zA-Z0-9 ,']+)\)", r'( \1 )', full_text)

    # Spacing the commas, dots, semicolons and colons.
    full_text = re.sub("([a-zA-Z0-9\"\'])([.,:;]) ", r"\1 \2 ", full_text)

    # Removing unicode chars (not accepted by SENNA SRL),
    # also removing first 2 and last char as they are markers (b' and ').
    full_text = str(unicodedata.normalize('NFD', full_text).encode('ascii', 'ignore'))[2:-1]

    # Removing backslashes.
    full_text = re.sub("\\\\", "", full_text)

    return full_text


# Remove stopwords and punctuation and stems a sentence.
def stem_and_stopword(sentence):
    stemmed_sent = []
    words = word_tokenize(sentence)
    filtered_words = [w for w in words if w not in stop_words]                                  # Removing stopwords.
    # Removing punctuation.
    filtered_words = [''.join(c for c in s if c not in string.punctuation) for s in filtered_words]
    filtered_words = [w for w in filtered_words if w]                                           # Removing blanks.

    for word in filtered_words:
        stemmed_sent.append(stemmer.stem(word))
    return stemmed_sent


# Remove punctuation.
def remove_punct(sentence):
    sent = ""
    words = sentence.split()
    filtered_words = [''.join(c for c in s if c not in string.punctuation) for s in words]      # Removing punctuation.
    filtered_words = [w for w in filtered_words if w]                                           # Removing blanks.

    for word in filtered_words:
        sent += word + " "
    return sent


# Compute the TFIDF score for each term in the sentences given the IDF of the dataset.
def tf_idf(sentences, idf_file_name):
    with open(idf_file_name, "rb") as fp:
        idfs = pickle.load(fp)
    max_idf = max(idfs.values())

    # Stemming the sentences to collect the terms.
    stems = []
    for sent in sentences:
        stems.extend(stem_and_stopword(sent))

    doc_dim = len(stems)
    terms = list(set(stems))

    tf_idfs = {}
    for term in terms:
        term_count = 0
        # Count how many time a term is present in the document.
        for stem in stems:
            if stem == term:
                term_count += 1
        term_f = term_count / doc_dim

        # Compute the TFIDF, if the idf is not present for that term the maximum value is used.
        if term in idfs.keys():
            tf_idfs[term] = term_f * idfs[term]
        else:
            tf_idfs[term] = term_f * max_idf

    # Normalizing the values by the maximum TFIDF value.
    max_tfidf = max(tf_idfs.values())
    for term in terms:
        tf_idfs[term] /= max_tfidf

    return tf_idfs


# Returns the sentence embeddings list of the input sentences list.
def sentence_embeddings(sentences):
    session = tf.Session()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddings = session.run(embedder(sentences))
    session.close()

    return embeddings


# Compute centrality scores of each sentence in a set of sentences (given as sentence embeddings).
# The score consist of the sum of the similarities between a sentence and all the others normalized by the max sum.
def centrality_scores(sent_embeddings):
    centr_scores = []
    for sent1 in sent_embeddings:
        sim_sum = 0
        for sent2 in sent_embeddings:
            sim_sum += np.inner(np.array(sent1), np.array(sent2))
        centr_scores.append(sim_sum)

    max_sum = max(centr_scores)
    centr_scores[:] = [x / max_sum for x in centr_scores]

    return centr_scores


# Using this to check the execution time.
def timer(text, start_time):
    current_time = time.time()
    print(text + str(current_time - start_time))
    return current_time


def tokens(text):
    return tokenizer.tokenize(text)


# Get the source text starting from the pas list (each pas contain the original sentence from which it has
# been extracted, there can be multiple pas with the same sentence).
def get_sources_from_pas_lists(pas_lists):
    sources = []
    for pas_list in pas_lists:
        sentences = []
        for pas in pas_list:
            if pas.sentence not in sentences:
                sentences.append(pas.sentence)

        ref = ""
        for sent in sentences:
            ref += sent + ". "

        sources.append(ref)

    return sources


def plot_history(model_name):
    with open(os.getcwd() + "/models/" + model_name + ".hst", "rb") as file:
        history = pickle.load(file)
    # Get training and test loss histories
    training_acc = history['acc']
    test_acc = history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_acc) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, test_acc, 'b-')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig(result_path + "histories/" + model_name + ".pdf")
    plt.close()


# Print some relevant summaries from a batch of texts.
def sample_summaries(model_name, docs_pas_lists, refs, recall_score_list, batch=-1, summaries=None):
    docs = get_sources_from_pas_lists(docs_pas_lists)
    if summaries is None:
        summaries = [generate_summary(model_name, pas_list) for pas_list in docs_pas_lists]

    rnd_index = random.randint(1, len(docs))
    best_index = recall_score_list.index(max(recall_score_list))
    worst_index = recall_score_list.index(min(recall_score_list))

    avg_recall_score = np.mean(recall_score_list)
    distances = [abs(avg_recall_score - score) for score in recall_score_list]
    avg_index = distances.index(min(distances))

    with open(result_path + "sample_summaries/" + model_name + "_sample_summaries.txt", "w") as dest_f:
        print("SAMPLES EXTRACTED USING MODEL:" + model_name, file=dest_f)
        if batch > -1:
            print("FROM BATCH: " + str(batch), file=dest_f)

        print("FIRST DOCUMENT:", file=dest_f)
        print("ORIGINAL DOCUMENT:", file=dest_f)
        print(docs[0], file=dest_f)
        print("REFERENCE:", file=dest_f)
        print(refs[0], file=dest_f)
        print("GENERATED SUMMARY:", file=dest_f)
        print(summaries[0], file=dest_f)
        print("ROUGE 1 RECALL: " + str(recall_score_list[0]), file=dest_f)

        print("RANDOM DOCUMENT (index: " + str(rnd_index) + "):", file=dest_f)
        print("ORIGINAL DOCUMENT:", file=dest_f)
        print(docs[rnd_index], file=dest_f)
        print("REFERENCE:", file=dest_f)
        print(refs[rnd_index], file=dest_f)
        print("GENERATED SUMMARY:", file=dest_f)
        print(summaries[rnd_index], file=dest_f)
        print("ROUGE 1 RECALL: " + str(recall_score_list[rnd_index]), file=dest_f)

        print("BEST RECALL DOCUMENT (index: " + str(best_index) + "):", file=dest_f)
        print("ORIGINAL DOCUMENT:", file=dest_f)
        print(docs[best_index], file=dest_f)
        print("REFERENCE:", file=dest_f)
        print(refs[best_index], file=dest_f)
        print("GENERATED SUMMARY:", file=dest_f)
        print(summaries[best_index], file=dest_f)
        print("ROUGE 1 RECALL: " + str(recall_score_list[best_index]), file=dest_f)

        print("WORST RECALL DOCUMENT (index: " + str(worst_index) + "):", file=dest_f)
        print("ORIGINAL DOCUMENT:", file=dest_f)
        print(docs[worst_index], file=dest_f)
        print("REFERENCE:", file=dest_f)
        print(refs[worst_index], file=dest_f)
        print("GENERATED SUMMARY:", file=dest_f)
        print(summaries[worst_index], file=dest_f)
        print("ROUGE 1 RECALL: " + str(recall_score_list[worst_index]), file=dest_f)

        print("AVERAGE RECALL DOCUMENT (index: " + str(avg_index) + "):", file=dest_f)
        print("ORIGINAL DOCUMENT:", file=dest_f)
        print(docs[avg_index], file=dest_f)
        print("REFERENCE:", file=dest_f)
        print(refs[avg_index], file=dest_f)
        print("GENERATED SUMMARY:", file=dest_f)
        print(summaries[avg_index], file=dest_f)
        print("ROUGE 1 RECALL: " + str(recall_score_list[avg_index]), file=dest_f)
