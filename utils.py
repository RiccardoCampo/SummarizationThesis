import os
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
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
embedder = hub.Module("https://tfhub.dev/google/random-nnlm-en-dim128/1")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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
