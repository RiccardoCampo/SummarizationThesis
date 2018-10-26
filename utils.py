import os
import random
import re
import string
import unicodedata
import time
import pickle
import corenlp

import nltk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

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


# Get the source text starting from the pas list (each pas contain the original sentence from which it has
# been extracted, there can be multiple pas with the same sentence).
def get_sources_from_pas_lists(pas_lists, dots=True):
    sources = []
    for pas_list in pas_lists:
        sentences = []
        for pas in pas_list:
            if pas.sentence not in sentences:
                sentences.append(pas.sentence)

        ref = ""
        for sent in sentences:
            if dots:
                ref += (sent + ". ").replace("..", ".")
            else:
                ref += (sent + "\n")

        sources.append(ref)

    return sources


# Print some relevant summaries from a batch of texts.
def sample_summaries(model_name, docs, refs, summaries, recall_score_list, batch=-1):
    best_index = recall_score_list.index(max(recall_score_list))
    worst_index = recall_score_list.index(min(recall_score_list))

    avg_recall_score = np.mean(recall_score_list)
    distances = [abs(avg_recall_score - score) for score in recall_score_list]
    avg_index = distances.index(min(distances))

    indices = [0, best_index, worst_index, avg_index]
    labels = ["FIRST", "BEST", "WORST", "AVERAGE"]

    for i in range(10):
        indices.append(random.randint(1, len(recall_score_list) - 1))
        labels.append("RANDOM")

    with open(os.getcwd() + "/results/sample_summaries/" +
              model_name + "_" + str(batch) +
              "_sample_summaries.txt", "w") as dest_f:
        print("SAMPLES EXTRACTED USING MODEL:" + model_name, file=dest_f)
        if batch > -1:
            print("FROM BATCH: " + str(batch), file=dest_f)

        for i in range(len(indices)):
            print(labels[i] + " DOCUMENT (index: " + str(indices[i]) + "): ", file=dest_f)
            print("ROUGE 1 RECALL: " + str(recall_score_list[indices[i]]), file=dest_f)
            print("ORIGINAL DOCUMENT:", file=dest_f)
            print(docs[indices[i]], file=dest_f)
            print("REFERENCE:", file=dest_f)
            print(refs[indices[i]], file=dest_f)
            print("GENERATED SUMMARY:", file=dest_f)
            print(summaries[indices[i]], file=dest_f)
            print("=================================", file=dest_f)


# Compute the ratio between direct speech in the text and the whole text (given a pas list).
def direct_speech_ratio(sentences):
    size = 0
    ds_size = 0
    used_sentences = []
    for sentence in sentences:
        if sentence not in used_sentences:
            used_sentences.append(sentence)

            trimmed_sentence = re.sub(
                '([a-zA-Z0-9 .,:;\'_\-]+)\"([a-zA-Z0-9 .,:;\'_\-]+)\"([a-zA-Z0-9 .,:;\'_\-]+)', r'\1 \3', sentence)
            trimmed_sentence = re.sub('\"([a-zA-Z0-9 .,:;\'_\-]+)\"([a-zA-Z0-9 .,:;\'_\-]+)', r'\2', trimmed_sentence)
            trimmed_sentence = re.sub('([a-zA-Z0-9 .,:;\'_\-]+)\"([a-zA-Z0-9 .,:;\'_\-]+)\"', r'\1', trimmed_sentence)
            trimmed_sentence = re.sub('\"([a-zA-Z0-9 .,:;\'_\-]+)\"', '', trimmed_sentence)

            size += len(sentence)
            ds_size += len(sentence) - len(trimmed_sentence)

    return ds_size / size


def resolve_anaphora(sentences):
    text_structure = [[word for word in sentence.split()] for sentence in sentences]

    text = ""
    for sentence in sentences:
        text += sentence + " "

    with corenlp.CoreNLPClient(annotators="coref".split(), timeout=1000000000) as client:
        annotations = client.annotate(text)

    sentences_annotations = []

    deleted_sentences_modifiers = []
    ds_mod = 0

    for sentence_annotations in annotations.sentence:
        if len(remove_punct(corenlp.to_text(sentence_annotations))) < 4:
            ds_mod += 1
            deleted_sentences_modifiers.append(ds_mod)
        else:
            sentences_annotations.append(sentence_annotations)
            deleted_sentences_modifiers.append(ds_mod)

    sentence_modifiers = [0] * len(sentences)
    for i in range(len(sentences)):
        sentence_annotations = sentences_annotations[i]
        if sentence_annotations.hasCorefMentionsAnnotation:
            for mention in sentence_annotations.mentionsForCoref:
                if mention.mentionType == "PRONOMINAL":
                  #  print("-_--_--------_----______-")
                  #  print(sentences[i])
                  #  print(corenlp.to_text(sentence_annotations))
                  #  print(text_structure[i])
                  #  print([token.word for token in sentence_annotations.token])
                  #  print("-_--_--------_----______-")
                    pronoun_id = mention.mentionID
                    # Sentence, Begin, End.
                    pronoun_sent_index = i - deleted_sentences_modifiers[i]

                    tks = list(sentence_annotations.token)
                    # print(tks)
                    punct_modifier = sum([1 for j in range(len(tks)) if tks[j].word in string.punctuation and
                                          j < mention.startIndex])

                   # print(punct_modifier)
                   # print(sentence_modifiers[i])
                    pronoun_begin_index = mention.startIndex + sentence_modifiers[i] - punct_modifier
                    pronoun_end_index = mention.endIndex + sentence_modifiers[i] - punct_modifier
                  #  print(pronoun_begin_index)
                  #  print(pronoun_end_index)
                    pronoun_chains = []
                    for chain in annotations.corefChain:
                        for chain_mention in chain.mention:
                            if chain_mention.mentionID == pronoun_id:
                                pronoun_chains.append(chain)
                    if pronoun_chains:
                        pronoun_chains.sort(key=lambda x: x.representative, reverse=True)

                        selected_chain = pronoun_chains[0]
                        for chain_mention in selected_chain.mention:
                            if chain_mention.mentionType == "PROPER" or chain_mention.mentionType == "NOMINAL":
                                reference_sent_index = chain_mention.sentenceIndex - \
                                                       deleted_sentences_modifiers[chain_mention.sentenceIndex]
                                reference_begin_index = chain_mention.beginIndex
                                reference_end_index = chain_mention.endIndex
                               # print("ref sent index {}".format(reference_sent_index))
                               # print("text struct len {}".format(len(text_structure)))
                               # print(text_structure[reference_sent_index])
                                text_structure[pronoun_sent_index][pronoun_begin_index:pronoun_end_index] = \
                                    text_structure[reference_sent_index][reference_begin_index:reference_end_index]
                                sentence_modifiers[i] += reference_end_index - reference_begin_index - 1
                                break

    result_sentences = []
    for sentence in text_structure:
        result_sentence = ""
        for word in sentence:
            result_sentence += word + " "
        result_sentences.append(result_sentence)

    return result_sentences


def resolve_anaphora_pas_list(pas_list):
    realized_pas_list = [(pas.realized_pas + "..\n").replace(" ..", "..") for pas in pas_list]
    resolved_sentences = resolve_anaphora(realized_pas_list)

    for pas in pas_list:
        pas.realized_pas = resolved_sentences[pas_list.index(pas)]