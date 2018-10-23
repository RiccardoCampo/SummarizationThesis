import math
import os
import pickle
import re
import xml.etree.ElementTree
import numpy as np
from time import time
from pas import extract_pas
from summarization import score_document, score_document_bestn
from utils import stem_and_stopword, text_cleanup, tokens, timer, sentence_embeddings, centrality_scores, tf_idf


# Compute idfs given a document list, storing them in the specified destination file.
def compute_idfs(doc_list, dest_path):
    docs_number = len(doc_list)
    stems = []
    doc_stems = {}

    for doc in doc_list:
        doc_index = doc_list.index(doc)
        doc_stems[doc_index] = []
        for sent in tokens(doc):
            doc_stems[doc_index].extend(stem_and_stopword(sent))
        stems.extend(doc_stems[doc_index])

    # Terms are the stems (taken only once) which appears in the document list.
    terms = list(set(stems))

    idfs = {}

    terms_dim = len(terms)
    term_index = 0
    for term in terms:
        term_index += 1
        term_count = 0
        # Counting how many documents contains the term.
        for doc in doc_list:
            if term in doc_stems[doc_list.index(doc)]:
                term_count += 1

        idf = math.log10(docs_number / term_count)
        idfs[term] = idf
        print("{:.3%}".format(term_index / terms_dim))

    with open(dest_path, "wb") as dest_file:
        pickle.dump(idfs, dest_file)


# Getting documents and respective summaries from DUC dataset from XML files.
def get_duc():
    duc_path = os.getcwd() + "/dataset/duc_source"
    docs = []
    summaries = []
    doc_names = []
    # Each directory contains more than one document.
    for dir_name in sorted(os.listdir(duc_path + "/docs")):
        for doc_name in sorted(os.listdir(duc_path + "/docs/" + dir_name)):
            doc_names.append(doc_name)
            # Docs from this newspaper are not a well-formed XML file, need to put apices in attributes "P".
            if doc_name.startswith("FBIS"):
                with open(duc_path + "/docs/" + dir_name + "/" + doc_name, "r+") as doc_f:
                    doc = doc_f.read()
                    doc = re.sub(" P=([0-9]+)", r' P="\1"', doc)
                with open(duc_path + "/docs/" + dir_name + "/" + doc_name, "w") as doc_f:
                    print(doc, file=doc_f)
            xml_doc = xml.etree.ElementTree.parse(duc_path + "/docs/" + dir_name + "/" + doc_name).getroot()
            doc = ""
            # Every file from different newspaper uses different tags for the title and body of the article.
            if doc_name.startswith("AP"):
                doc = (xml_doc.find("HEAD").text + "." + xml_doc.find("TEXT").text) \
                    if xml_doc.find("HEAD") is not None else xml_doc.find("TEXT").text

            if doc_name.startswith("WSJ"):
                doc = xml_doc.find("HL").text + "." + xml_doc.find("TEXT").text

            if doc_name.startswith("FT") or doc_name.startswith("SJMN"):
                doc = xml_doc.find("HEADLINE").text + "." + xml_doc.find("TEXT").text

            if doc_name.startswith("FBIS"):
                doc = xml_doc.find("H3").find("TI").text + "." + xml_doc.find("TEXT").text

            if doc_name.startswith("LA"):
                for hl in xml_doc.find("HEADLINE").findall("P"):
                    doc += hl.text.replace("\n", "")
                doc += "."
                for tx in xml_doc.find("TEXT").findall("P"):
                    doc += tx.text
            doc.replace("\n", " ")
            docs.append(doc)

            # Getting the respective summary.
            found = False
            summ_dir_name = ""
            for summ_dir in sorted(os.listdir(duc_path + "/summaries")):
                if summ_dir.startswith(dir_name):
                    summ_dir_name = summ_dir
                    found = True
                    break
            if not found:
                print("ERROR: dir: " + dir_name + "not found in the summaries")

            with open(duc_path + "/summaries/" + summ_dir_name + "/perdocs", "r+") as doc_f:
                doc = doc_f.read()
                if not doc.startswith("<DOC>"):
                    doc = "<DOC>" + doc + "</DOC>"
                    doc_f.close()
                    with open(duc_path + "/summaries/" + summ_dir_name + "/perdocs", "w") as doc_f_w:
                        print(doc, file=doc_f_w)
            xml_doc = xml.etree.ElementTree.parse(duc_path + "/summaries/" + summ_dir_name + "/perdocs").getroot()
            for elem in xml_doc.findall("SUM"):
                if elem.get("DOCREF") == doc_name:
                    summaries.append(elem.text.replace("\n", " "))

    return docs, summaries, doc_names


# Load all the DUC documents and summaries, process them and store them.
def store_pas_duc_dataset():
    docs_pas_lists = []
    refs_pas_lists = []

    docs, references, _ = get_duc()
    # For each document the pas_list is extracted after cleaning the text and tokenizing it.
    for doc in docs:
        print("Processing doc " + str(docs.index(doc)) + "/" + str(len(docs)))
        doc = text_cleanup(doc)
        # Splitting sentences (by dot).
        sentences = tokens(doc)
        pas_list = extract_pas(sentences, "duc")
        docs_pas_lists.append(pas_list)

    # The list of pas lists is then stored.
    with open(os.getcwd() + "/dataset/duc/duc_docs_pas.dat", "wb") as dest_f:
        pickle.dump(docs_pas_lists, dest_f)
        
    # Same for reference summaries...
    for ref in references:
        print("Processing doc " + str(references.index(ref)) + "/" + str(len(references)))
        ref = text_cleanup(ref)
        # Splitting sentences (by dot).
        sentences = tokens(ref)
        pas_list = extract_pas(sentences, "duc", keep_all=True)
        refs_pas_lists.append(pas_list)

    with open(os.getcwd() + "/dataset/duc/duc_refs_pas.dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)


# Retrieve New York Times corpus documents and summaries.
def get_nyt(nyt_path, min_doc=0, max_doc=100):
    docs = []
    summaries = []
    count = 0

    # Dataset is divided by year, month day.
    for year_dir in sorted(os.listdir(nyt_path)):
        for month_dir in sorted(os.listdir(nyt_path + "/" + year_dir)):
            for day_dir in sorted(os.listdir(nyt_path + "/" + year_dir + "/" + month_dir + "/" + month_dir)):
                for doc_name in sorted(os.listdir(nyt_path + "/" + year_dir + "/" +
                                                  month_dir + "/" + month_dir + "/" + day_dir)):
                    # Extracting the documents between min_doc and max_doc.
                    if min_doc <= count < max_doc:
                        xml_doc = xml.etree.ElementTree.parse(nyt_path + "/" + year_dir + "/" +
                                                              month_dir + "/" + month_dir + "/" +
                                                              day_dir + "/" + doc_name).getroot()
                        doc = ""
                        # Adding the headline as first sentence.
                        doc += xml_doc.find("body/body.head/hedline/hl1").text + ".\n"
                        # Adding all the paragraphs in the block with the attribute "full_text"
                        for paragraph in xml_doc.findall("body/body.content/block[@class='full_text']/p"):
                            doc += paragraph.text + ".\n"

                        summary = xml_doc.find("body/body.head/abstract/p").text
                        # Check if the summary is non-empty and it is longer than 600.
                        if summary:
                            if len(summary) > 600:
                                docs.append(doc)
                                summaries.append(summary)
                                print(str(count) + ") " + year_dir + "|" + month_dir + "|" + day_dir + "|" + doc_name)
                    count += 1
                    if count >= max_doc:
                        break
                if count >= max_doc:
                    break
            if count >= max_doc:
                break
        if count >= max_doc:
            break
    return docs, summaries


# Load NYT documents and summaries, process them and store them.
# Process a number of documents between min_pas and max_pas.
def store_pas_nyt_dataset(nyt_path, min_pas, max_pas):
    docs_pas_lists = []
    refs_pas_lists = []

    docs, references = get_nyt(nyt_path, min_pas, max_pas)

    for i in range(len(docs)):
        start_time = time()
        print("Processing doc " + str(i) + "/" + str(len(docs)))
        doc = docs[i]
        ref = references[i]

        doc = text_cleanup(doc)
        # Splitting sentences (by dot).
        sentences = tokens(doc)
        doc_pas_list = extract_pas(sentences, "nyt")

        ref = text_cleanup(ref)
        # Splitting sentences (by dot).
        sentences = tokens(ref)
        ref_pas_list = extract_pas(sentences, "nyt", keep_all=True)

        if len(doc_pas_list) > 5 and len(doc_pas_list) >= len(ref_pas_list):
            refs_pas_lists.append(ref_pas_list)
            docs_pas_lists.append(doc_pas_list)
        timer(str(i) + " processed in:", start_time)

    # PAS lists are stored.
    with open(os.getcwd() + "/dataset/nyt/nyt_refs" + str(min_pas) + "-" + str(max_pas) + "_pas.dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)
    with open(os.getcwd() + "/dataset/nyt/nyt_docs" + str(min_pas) + "-" + str(max_pas) + "_pas.dat", "wb") as dest_f:
        pickle.dump(docs_pas_lists, dest_f)


# Putting pas lists in files of fixed length and without excessively long documents.
def arrange_nyt_pas_lists(dim=1000, max_len=300, max_file=660000):
    end_of_docs = False
    batch = 0
    while not end_of_docs:
        # Setting start and end index of the pas to compact.
        start = dim * batch
        end = start + dim
        print("Batch " + str(batch) + " (" + str(start) + "-" + str(end) + ")")

        docs_pas_lists = []
        refs_pas_lists = []

        docs_no = 0
        for i in range(0, max_file, 10000):
            # Loading documents until the end is reached.
            if docs_no < end:
                with open(os.getcwd() + "/dataset/nyt/raw/nyt_docs" + str(i) + "-" + str(i + 10000) + "_pas.dat",
                          "rb") as docs_f:
                    temp_docs = pickle.load(docs_f)
                with open(os.getcwd() + "/dataset/nyt/raw/nyt_refs" + str(i) + "-" + str(i + 10000) + "_pas.dat",
                          "rb") as refs_f:
                    temp_refs = pickle.load(refs_f)
                temp_docs_len = len(temp_docs)
                # Clear the documents with exceeding length and respective summaries.
                j = 0
                while j < temp_docs_len:
                    if len(temp_docs[j]) > max_len or \
                            len(temp_docs[j]) < len(temp_refs[j]) or \
                            len(temp_refs[j]) <= 0:
                        del temp_refs[j]
                        del temp_docs[j]
                    else:
                        j += 1
                    temp_docs_len = len(temp_docs)

                # Of the current file retain the documents from the starting point until the end is reached.
                if docs_no + temp_docs_len > start:
                    print("file " + str(i) + " with dim: " + str(temp_docs_len))
                    print("used from " + str(max(start - docs_no, 0)) + " to " + str(min(end - docs_no, temp_docs_len)))

                    temp_docs = temp_docs[max(start - docs_no, 0): min(end - docs_no, temp_docs_len)]
                    temp_refs = temp_refs[max(start - docs_no, 0): min(end - docs_no, temp_docs_len)]
                    temp_docs_len = len(temp_docs) + max(start - docs_no, 0)

                    docs_pas_lists.extend(temp_docs)
                    refs_pas_lists.extend(temp_refs)

                    print("batch " + str(batch) + " has now " + str(docs_no + temp_docs_len - start) + " documents")
                docs_no += temp_docs_len
            else:
                if i == max_file - 10000:
                    end_of_docs = True
                break

        with open(os.getcwd() + "/dataset/nyt/compact/compact_nyt_docs_pas" + str(batch) + ".dat", "wb") as dest_f:
            pickle.dump(docs_pas_lists, dest_f)
        with open(os.getcwd() + "/dataset/nyt/compact/compact_nyt_refs_pas" + str(batch) + ".dat", "wb") as dest_f:
            pickle.dump(refs_pas_lists, dest_f)

        batch += 1


# Getting the pas lists of documents and reference summaries. The index represent the batch of compact nyt pas to get.
# Or, if -1, it tells to get duc pas lists.
def get_pas_lists(index=-1):
    if index < 0:
        with open(os.getcwd() + "/dataset/duc/duc_docs_pas.dat", "rb") as docs_f:
            docs_pas_lists = pickle.load(docs_f)
        with open(os.getcwd() + "/dataset/duc/duc_refs_pas.dat", "rb") as refs_f:
            refs_pas_lists = pickle.load(refs_f)
    else:
        with open(os.getcwd() + "/dataset/nyt/compact/compact_nyt_docs_pas" + str(index) + ".dat", "rb") as docs_f:
            docs_pas_lists = pickle.load(docs_f)
        with open(os.getcwd() + "/dataset/nyt/compact/compact_nyt_refs_pas" + str(index) + ".dat", "rb") as refs_f:
            refs_pas_lists = pickle.load(refs_f)
    return docs_pas_lists, refs_pas_lists


# Matrix representation is computed and stored.
def store_matrices(index):
    if index < 0:
        docs_pas_lists, refs_pas_lists = get_pas_lists(-1)
        dataset_path = "/dataset/duc/duc"
    else:
        docs_pas_lists, refs_pas_lists = get_pas_lists(index)
        dataset_path = "/dataset/nyt/" + str(index) + "/nyt" + str(index)

    # Storing the matrices in the appropriate file, depending on the scoring system.
    doc_path = dataset_path + "_doc_matrix.dat"
    ref_path = dataset_path + "_ref_matrix.dat"

    docs_no = len(docs_pas_lists)                                   # First dimension, documents number.
    # Second dimension, max document length (sparse), fixed in case of nyt.
    max_sent_no = max([len(doc) for doc in docs_pas_lists]) if index < 0 else 300
    # Third dimension, vector representation dimension.
    sent_vec_len = len(docs_pas_lists[0][0].vector) + len(docs_pas_lists[0][0].embeddings)

    # The matrix are initialized as zeros, then they'll filled in with vectors for each docs' sentence.
    refs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))
    docs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))

    for i in range(docs_no):
        for j in range(max_sent_no):
            if j < len(docs_pas_lists[i]):
                docs_3d_matrix[i, j, :] = np.append(docs_pas_lists[i][j].vector, docs_pas_lists[i][j].embeddings)
            if j < len(refs_pas_lists[i]):
                refs_3d_matrix[i, j, :] = np.append(refs_pas_lists[i][j].vector, refs_pas_lists[i][j].embeddings)

    with open(os.getcwd() + ref_path, "wb") as dest_f:
        pickle.dump(refs_3d_matrix, dest_f)
    with open(os.getcwd() + doc_path, "wb") as dest_f:
        pickle.dump(docs_3d_matrix, dest_f)


# Scores are computed and stored.
def store_score_matrices(index, binary_scores):
    if index < 0:
        dataset_path = "/dataset/duc/duc"
    else:
        dataset_path = "/dataset/nyt/" + str(index) + "/nyt" + str(index)

    # Storing the scores in the appropriate file, depending on the scoring system.
    doc_path = dataset_path + "_doc_matrix.dat"
    ref_path = dataset_path + "_ref_matrix.dat"

    with open(os.getcwd() + ref_path, "rb") as dest_f:
        refs_3d_matrix = pickle.load(dest_f)
    with open(os.getcwd() + doc_path, "rb") as dest_f:
        docs_3d_matrix = pickle.load(dest_f)

    docs_no = docs_3d_matrix.shape[0]
    max_sent_no = docs_3d_matrix.shape[1]

    weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                    (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                    (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]

    for weights in weights_list:
        if binary_scores:
            scores_path = dataset_path + "_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + "binary.dat"
        else:
            scores_path = dataset_path + "_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + ".dat"

        # Build the score matrix document by document.
        scores_matrix = np.zeros((docs_no, max_sent_no))
        for i in range(docs_no):
            scores_matrix[i] = score_document(docs_3d_matrix[i, :, :], refs_3d_matrix[i, :, :],
                                              weights, binary=binary_scores)
        with open(os.getcwd() + scores_path, "wb") as dest_f:
            pickle.dump(scores_matrix, dest_f)


# Second binary scoring method.
def store_bestn_score_matrices(index):
    if index < 0:
        dataset_path = "/dataset/duc/duc"
    else:
        dataset_path = "/dataset/nyt/" + str(index) + "/nyt" + str(index)

    # Storing the matrices in the appropriate file, depending on the scoring system.
    doc_path = dataset_path + "_doc_matrix.dat"
    ref_path = dataset_path + "_ref_matrix.dat"

    with open(os.getcwd() + ref_path, "rb") as dest_f:
        refs_3d_matrix = pickle.load(dest_f)
    with open(os.getcwd() + doc_path, "rb") as dest_f:
        docs_3d_matrix = pickle.load(dest_f)

    docs_no = docs_3d_matrix.shape[0]
    max_sent_no = docs_3d_matrix.shape[1]

    scores_path = dataset_path + "_score_matrix_bestn.dat"

    scores_matrix = np.zeros((docs_no, max_sent_no))
    for i in range(docs_no):
        scores_matrix[i] = score_document_bestn(docs_3d_matrix[i, :, :], refs_3d_matrix[i, :, :])
    with open(os.getcwd() + scores_path, "wb") as dest_f:
        pickle.dump(scores_matrix, dest_f)


# Getting the matrices of documents and reference summaries.
def get_matrices(weights, scores, index=-1):
    # Selecting the right path depending on the batch or binary scoring.
    if index < 0:
        dataset_path = "/dataset/duc/duc"
    else:
        dataset_path = "/dataset/nyt/" + str(index) + "/nyt" + str(index)

    if scores == 1:
        # scores_path = dataset_path + "_sent_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + "binary.dat"
        scores_path = dataset_path + "_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + "binary.dat"
    elif scores == 2:
        #scores_path = dataset_path + "_sent_score_matrix_bestn.dat"
        scores_path = dataset_path + "_score_matrix_bestn.dat"
    else:
        #scores_path = dataset_path + "_sent_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + ".dat"
        scores_path = dataset_path + "_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + ".dat"

    # doc_path = dataset_path + "_doc_sent_matrix.dat"
    doc_path = dataset_path + "_doc_matrix.dat"
    # ref_path = dataset_path + "_ref_sent_matrix.dat"
    ref_path = dataset_path + "_ref_matrix.dat"

    with open(os.getcwd() + doc_path, "rb") as docs_f:
        doc_matrix = pickle.load(docs_f)
    with open(os.getcwd() + ref_path, "rb") as refs_f:
        ref_matrix = pickle.load(refs_f)
    with open(os.getcwd() + scores_path, "rb") as scores_f:
        score_matrix = pickle.load(scores_f)

    return doc_matrix, ref_matrix, score_matrix


def store_full_sentence_matrices():
    docs, references, _ = get_duc()

    docs_no = len(docs)                                   # First dimension, documents number.
    # Second dimension, max document length (sparse), fixed in case of nyt.
    max_sent_no = 200
    # Third dimension, vector representation dimension.
    sent_vec_len = 134

    # The matrix are initialized as zeros, then they'll filled in with vectors for each docs' sentence.
    refs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))
    docs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))

    # For each document the pas_list is extracted after cleaning the text and tokenizing it.
    for k in range(2):
        if k == 0:
            doc_list = docs
        else:
            doc_list = references

        for i in range(len(doc_list)):
            doc = doc_list[i]
            print("Processing doc " + str(i) + "/" + str(len(docs)))
            doc = text_cleanup(doc)
            # Splitting sentences (by dot).
            sentences = tokens(doc)
            embeddings = sentence_embeddings(sentences)
            centr_scores = centrality_scores(embeddings)
            tf_idfs = tf_idf(sentences, os.getcwd() + "/dataset/duc/duc_idfs.dat")
            # Position score, reference sentence length score, tf_idf, numerical data, centrality, title.
            for j in range(len(sentences)):
                sent = sentences[j]

                position_score = (len(sentences) - j) / len(sentences)
                length_score = len(sent) / max(len(snt) for snt in sentences)
                tf_idf_score = 0
                numerical_score = 0
                centrality_score = centr_scores[j]
                title_sim_score = np.inner(np.array(embeddings[j]), np.array(embeddings[-1]))

                # Computing centrality and tf_idf score.
                terms = list(set(stem_and_stopword(sent)))
                for term in terms:
                    # Due to errors terms may be not present in the tf_idf dictionary.
                    if term in tf_idfs.keys():
                        tf_idf_score += tf_idfs[term]
                    else:
                        tf_idf_score += 0

                    if term.isdigit():
                        numerical_score += 1

                # Some errors in the preprocessing may lead to zero terms, so it is necessary to avoid division by zero.
                if len(terms):
                    tf_idf_score /= len(terms)
                else:
                    tf_idf_score = 0

                if k ==0:
                    docs_3d_matrix[i, j, :] = np.append([position_score, length_score,
                                                         tf_idf_score, numerical_score,
                                                         centrality_score, title_sim_score], embeddings[j])
                else:
                    refs_3d_matrix[i, j, :] = np.append([position_score, length_score,
                                                         tf_idf_score, numerical_score,
                                                         centrality_score, title_sim_score], embeddings[j])

    # Storing the matrices in the appropriate file, depending on the scoring system.
    doc_path = "/dataset/duc/duc_doc_sent_matrix.dat"
    ref_path = "/dataset/duc/duc_ref_sent_matrix.dat"

    with open(os.getcwd() + ref_path, "wb") as dest_f:
        pickle.dump(refs_3d_matrix, dest_f)
    with open(os.getcwd() + doc_path, "wb") as dest_f:
        pickle.dump(docs_3d_matrix, dest_f)
