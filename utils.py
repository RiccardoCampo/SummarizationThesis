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
anaphora = corenlp.CoreNLPClient(annotators="coref".split(), timeout=1000000000)
# anaphora = None


def text_cleanup(full_text):
    """
    Preprocessing the input text to make it ready for the SRL.

    :param full_text: text input.
    :return: processed text.
    """
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

    # Removing unicode chars (not accepted by SENNA SRL),
    # also removing first 2 and last char as they are markers (b' and ').
    full_text = str(unicodedata.normalize('NFD', full_text).encode('ascii', 'ignore'))[2:-1]

    # Removing backslashes.
    full_text = re.sub("\\\\", "", full_text)

    return full_text


def stem_and_stopword(sentence):
    """
    Remove stopwords and punctuation and stems a sentence.

    :param sentence: input sentence.
    :return: list of stems.
    """
    stemmed_sent = []
    words = word_tokenize(sentence)
    filtered_words = [w for w in words if w not in stop_words]                                  # Removing stopwords.
    # Removing punctuation.
    filtered_words = [''.join(c for c in s if c not in string.punctuation) for s in filtered_words]
    filtered_words = [w for w in filtered_words if w]                                           # Removing blanks.

    for word in filtered_words:
        stemmed_sent.append(stemmer.stem(word))
    return stemmed_sent


def remove_punct(sentence):
    """
    Remove punctuation.

    :param sentence: input sentence.
    :return: sentence without punctuation.
    """
    sent = ""
    words = sentence.split()
    filtered_words = [''.join(c for c in s if c not in string.punctuation) for s in words]      # Removing punctuation.
    filtered_words = [w for w in filtered_words if w]                                           # Removing blanks.

    for word in filtered_words:
        sent += word + " "
    return sent


def tf_idf(sentences, idf_file_name):
    """
    Compute the TFIDF score for each term in the sentences given the IDF of the dataset.

    :param sentences: list of sentences.
    :param idf_file_name: name of the idf file.
    :return: tf-idf scores of the terms in the input sentences.
    """
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


def sentence_embeddings(sentences):
    """
    Returns the sentence embeddings list of the input sentences list.

    :param sentences: list of sentences.
    :return: list of sentence embeddings.
    """
    session = tf.Session()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddings = session.run(embedder(sentences))
    session.close()

    return embeddings


def centrality_scores(sent_embeddings):
    """
    Compute centrality scores of each sentence in a set of sentences (given as sentence embeddings).
    The score consist of the sum of the similarities between a sentence and all the others normalized by the max sum.

    :param sent_embeddings: list of sentences' embeddings.
    :return: centrality score of each sentence.
    """
    centr_scores = []
    for sent1 in sent_embeddings:
        sim_sum = 0
        for sent2 in sent_embeddings:
            sim_sum += np.inner(np.array(sent1), np.array(sent2))
        centr_scores.append(sim_sum)

    max_sum = max(centr_scores)
    centr_scores[:] = [x / max_sum for x in centr_scores]

    return centr_scores


def timer(text, start_time):
    """
    Using this to check the execution time.

    :param text: string to print.
    :param start_time: start time.
    :return: current time.
    """
    current_time = time.time()
    print(text + str(current_time - start_time))
    return current_time


def tokens(text):
    """
    Tokenize the input text.

    :param text: input text.
    :return: list of sentences.
    """
    return tokenizer.tokenize(text)


def get_sources_from_pas_lists(pas_lists, dots=True):
    """
    Get the source text starting from the pas list (each pas contain the original sentence from which it has
    been extracted, there can be multiple pas with the same sentence).

    :param pas_lists: list of pas list.
    :param dots: if True add dots at the end of each sentence.
    :return: list of source documents.
    """
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


def sample_summaries(model_name, docs, refs, summaries, recall_score_list, batch=-1, all_summ=False):
    """
    Print some relevant summaries from a batch of texts.

    :param model_name: name of the model used to produce the summaries.
    :param docs: list of original documents.
    :param refs: list of reference summaries.
    :param summaries: list of system generated summaries.
    :param recall_score_list: list of rouge 1 recall score of each summary.
    :param batch: batch number.
    :param all_summ: stores all the summaries, not just 10.
    """
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

        if all_summ:
            indices = range(len(recall_score_list))
            labels = [""] * len(recall_score_list)

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


def direct_speech_ratio(sentences):
    """
    Compute the ratio between direct speech in the text and the whole text.

    :param sentences: list of sentences.
    :return: direct speech ratio of the document.
    """
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


def direct_speech_ratio_pas(pas_list):
    """
    Compute the ratio between direct speech in the text and the whole text length given a pas list.

    :param pas_list: list of pas.
    :return: direct speech ratio of the document.
    """
    size = 0
    ds_size = 0
    used_sentences = []
    for pas in pas_list:
        sentence = pas.sentence
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
    """
    Resolve anaphora (pronominal coreference).

    :param sentences: list of input sentences.
    :return: sentences with resolved pronouns.
    """
    # I'm using a matrix to represent the text, each row is a sentence each column a word
    text_structure = [[word for word in sentence.split()] for sentence in sentences]

    # Putting together all the sentence in the same text.
    text = ""
    for sentence in sentences:
        text += sentence + " "

    annotations = anaphora.annotate(text)
    sentences_annotations = []

    # I remove punctuation as it is considered as a token by coref but not in the text structure, I take notice of how
    # many punctuation signs I removed to then correct the indexing of the text structure.
    deleted_sentences_modifiers = []
    ds_mod = 0

    for sentence_annotations in annotations.sentence:
        if len(remove_punct(corenlp.to_text(sentence_annotations))) < 2:
            ds_mod += 1
            deleted_sentences_modifiers.append(ds_mod)
        else:
            sentences_annotations.append(sentence_annotations)
            deleted_sentences_modifiers.append(ds_mod)

    # If a pronoun is substituted by more than a word I need to take into account that the indexing will vary.
    sentence_modifiers = [0] * len(sentences)

    # For each sentence, if it has annotations, for each pronominal mention I perform the resolution.
    for i in range(len(sentences)):
        sentence_annotations = sentences_annotations[i]
        if sentence_annotations.hasCorefMentionsAnnotation:
            for mention in sorted(sentence_annotations.mentionsForCoref, key=lambda x: x.startIndex):
                if mention.mentionType == "PRONOMINAL":
                    pronoun_id = mention.mentionID
                    pronoun_sent_index = i

                    tks = list(sentence_annotations.token)
                    punct_modifier = sum([1 for j in range(len(tks)) if tks[j].word in string.punctuation and
                                          j < mention.startIndex])

                    pronoun_begin_index = mention.startIndex + sentence_modifiers[i] - punct_modifier
                    pronoun_end_index = mention.endIndex + sentence_modifiers[i] - punct_modifier

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
    """
    Resolve the anaphora starting from a pas_list. Directly modifies the realized_pas field in each pas.

    :param pas_list: pas list.
    """
    realized_pas_list = [(pas.realized_pas
                          .replace("...", "")
                          .replace("..", "") + "..\n").replace(" ..", "..").replace("...", "..") for pas in pas_list]
    resolved_sentences = resolve_anaphora(realized_pas_list)

    for pas in pas_list:
        pas.realized_pas = resolved_sentences[pas_list.index(pas)]
