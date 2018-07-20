import math
import os
import pickle
import re
import xml.etree.ElementTree
import numpy as np
from numpy.random.mtrand import permutation

from pas import extract_pas
from summarization import score_document
from utils import stem_and_stopword, text_cleanup, tokens


# Compute idfs given a document list, storing them in the specified destination file.
def compute_idfs(doc_list, dest_file):
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

    pickle.dump(idfs, dest_file)


# Getting documents and respective summaries from DUC dataset from XML files.
def get_duc(duc_path, batch=0):
    docs = []
    summaries = []
    doc_names = []
    # Each directory contains more than one document.
    for dir_name in os.listdir(duc_path + "/docs"):
        for doc_name in os.listdir(duc_path + "/docs/" + dir_name):
            doc_names.append(doc_name)
            # Docs from this newspaper are not a well-formed XML file, need to put apices in attributes "P".
            if doc_name.startswith("FBIS"):
                with open(duc_path + "/docs/" + dir_name + "/" + doc_name, "r+") as doc_f:
                    doc = doc_f.read()
                    doc = re.sub(" P=([0-9]+)", r' P="\1"', doc)
                    doc_f.seek(0)
                    doc_f.write(doc)
                    doc_f.close()

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
            for summ_dir in os.listdir(duc_path + "/summaries"):
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
                doc_f.seek(0)
                doc_f.write(doc)
                doc_f.close()
            xml_doc = xml.etree.ElementTree.parse(duc_path + "/summaries/" + summ_dir_name + "/perdocs").getroot()
            for elem in xml_doc.findall("SUM"):
                if elem.get("DOCREF") == doc_name:
                    summaries.append(elem.text.replace("\n", " "))

    # Shuffles the documents loading the indices.
    if batch:
        with open(os.getcwd() + "/dataset/batch" + str(batch) + "/indexes.dat", "rb") as docs_f:
            indexes = pickle.load(docs_f)
        summaries = [summaries[i] for i in indexes]
        docs = [docs[i] for i in indexes]
        doc_names = [doc_names[i] for i in indexes]

    return docs, summaries, doc_names


# Load all the DUC documents and summaries, process them and store them.
def store_pas_duc_dataset(duc_path):
    docs_pas_lists = []
    refs_pas_lists = []

    docs, references, names = get_duc(duc_path)
    # For each document the pas_list is extracted after cleaning the text and tokenizing it.
    for doc in docs:
        print("Processing doc " + str(docs.index(doc)) + "/" + str(len(docs)))
        doc = text_cleanup(doc)
        # Splitting sentences (by dot).
        sentences = tokens(doc)
        pas_list = extract_pas(sentences, "duc")
        docs_pas_lists.append(pas_list)

    # The list of pas lists is then stored.
    with open(os.getcwd() + "/dataset/duc_docs_pas.dat", "wb") as dest_f:
        pickle.dump(docs_pas_lists, dest_f)
        
    # Same for reference summaries...
    for ref in references:
        print("Processing doc " + str(references.index(ref)) + "/" + str(len(references)))
        ref = text_cleanup(ref)
        # Splitting sentences (by dot).
        sentences = tokens(ref)
        pas_list = extract_pas(sentences, "duc", keep_all=True)
        refs_pas_lists.append(pas_list)

    with open(os.getcwd() + "/dataset/duc_refs_pas.dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)


# Getting the pas lists of documents and reference summaries.
def get_pas_lists(batch=0):
    dataset_path = "/dataset"
    if batch:
        dataset_path = "/dataset/batch" + str(batch)
    doc_path = dataset_path + "/duc_docs_pas.dat"
    ref_path = dataset_path + "/duc_refs_pas.dat"

    with open(os.getcwd() + doc_path, "rb") as docs_f:
        docs_pas_lists = pickle.load(docs_f)
    with open(os.getcwd() + ref_path, "rb") as refs_f:
        refs_pas_lists = pickle.load(refs_f)

    return docs_pas_lists, refs_pas_lists


# Matrix representation is computed and stored.
def store_duc_matrices(weights):
    docs_pas_lists, refs_pas_lists = get_pas_lists()

    docs_no = len(docs_pas_lists)                                   # First dimension, documents number.
    max_sent_no = max([len(doc) for doc in docs_pas_lists])         # Second dimension, max document length (sparse).
    # Third dimension, vector representation dimension.
    sent_vec_len = len(docs_pas_lists[0][0].vector) + len(docs_pas_lists[0][0].embeddings)

    # The matrix are initialized as zeros, then they'll filled in with vectors for each docs' sentence.
    docs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))
    refs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))

    for i in range(docs_no):
        for j in range(max_sent_no):
            if j < len(docs_pas_lists[i]):
                docs_3d_matrix[i, j, :] = np.append(docs_pas_lists[i][j].vector, docs_pas_lists[i][j].embeddings)
            if j < len(refs_pas_lists[i]):
                refs_3d_matrix[i, j, :] = np.append(refs_pas_lists[i][j].vector, refs_pas_lists[i][j].embeddings)

    # Storing the matrices in the appropriate file, depending on dataset and embedding.
    doc_path = "/dataset/duc_doc_matrix.dat"
    ref_path = "/dataset/duc_ref_matrix.dat"
    scores_path = "/dataset/duc_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + ".dat"

    with open(os.getcwd() + doc_path, "wb") as dest_f:
        pickle.dump(docs_3d_matrix, dest_f)
    with open(os.getcwd() + ref_path, "wb") as dest_f:
        pickle.dump(refs_3d_matrix, dest_f)

    scores_matrix = np.zeros((docs_no, max_sent_no))
    for i in range(docs_no):
        scores_matrix[i] = score_document(docs_3d_matrix[i, :, :], refs_3d_matrix[i, :, :], weights)
    with open(os.getcwd() + scores_path, "wb") as dest_f:
        pickle.dump(scores_matrix, dest_f)


# Getting the matrices of documents and reference summaries.
def get_matrices(weights=(0.5, 0.5), batch=0):
    dataset_path = "/dataset"
    if batch:
        dataset_path = "/dataset/batch" + str(batch)
    doc_path = dataset_path + "/duc_doc_matrix.dat"
    ref_path = dataset_path + "/duc_ref_matrix.dat"
    scores_path = dataset_path + "/duc_score_matrix" + str(weights[0]) + "-" + str(weights[1]) + ".dat"

    with open(os.getcwd() + doc_path, "rb") as docs_f:
        doc_matrix = pickle.load(docs_f)
    with open(os.getcwd() + ref_path, "rb") as refs_f:
        ref_matrix = pickle.load(refs_f)
    with open(os.getcwd() + scores_path, "rb") as refs_f:
        score_matrix = pickle.load(refs_f)

    return doc_matrix, ref_matrix, score_matrix


# Shuffles the matrices and pas lists with a random permutation (and stores the permutation)
def shuffle_data(batch):
    docs_pas_lists, refs_pas_lists = get_pas_lists()
    doc_matrix, ref_matrix, _ = get_matrices()

    batch_path = "/dataset/batch" + str(batch)

    # Compute and store permutation.
    dataset_size = len(docs_pas_lists)
    indexes = permutation(dataset_size)
    with open(os.getcwd() + batch_path + "/indexes.dat", "wb") as dest_f:
        pickle.dump(indexes, dest_f)

    # Shuffle and store pas lists.
    docs_pas_lists = [docs_pas_lists[i] for i in indexes]
    refs_pas_lists = [refs_pas_lists[i] for i in indexes]
    with open(os.getcwd() + batch_path + "/duc_docs_pas.dat", "wb") as dest_f:
        pickle.dump(docs_pas_lists, dest_f)
    with open(os.getcwd() + batch_path + "/duc_refs_pas.dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)

    # Shuffle and store matrices
    shuffled_doc_matrix = np.zeros(doc_matrix.shape)
    shuffled_ref_matrix = np.zeros(ref_matrix.shape)
    for i in range(dataset_size):
        shuffled_doc_matrix[i, :, :] = doc_matrix[indexes[i], :, :]
        shuffled_ref_matrix[i, :, :] = ref_matrix[indexes[i], :, :]
    with open(os.getcwd() + batch_path + "/duc_doc_matrix.dat", "wb") as dest_f:
        pickle.dump(shuffled_doc_matrix, dest_f)
    with open(os.getcwd() + batch_path + "/duc_ref_matrix.dat", "wb") as dest_f:
        pickle.dump(shuffled_ref_matrix, dest_f)

    # Same for every score matrix.
    weights_list = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
                    (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
                    (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    for weights in weights_list:
        _, _, scores_matrix = get_matrices(weights=weights)
        shuffled_scores_matrix = np.zeros(scores_matrix.shape)
        for i in range(dataset_size):
            shuffled_scores_matrix[i, :] = scores_matrix[indexes[i], :]
        with open(os.getcwd() + batch_path + "/duc_score_matrix" + str(weights[0]) + "-" +
                  str(weights[1]) + ".dat", "wb") as dest_f:
            pickle.dump(shuffled_scores_matrix, dest_f)


# Retrieve New York Times corpus documents and summaries.
def get_nyt(nyt_path, min_doc=0, max_doc=100):
    docs = []
    summaries = []
    count = 0

    # Dataset is divided by year, month day.
    for year_dir in os.listdir(nyt_path):
        for month_dir in os.listdir(nyt_path + "/" + year_dir):
            print(year_dir + month_dir)
            for day_dir in os.listdir(nyt_path + "/" + year_dir + "/" + month_dir + "/" + month_dir):
                for doc_name in os.listdir(nyt_path + "/" + year_dir + "/" +
                                           month_dir + "/" + month_dir + "/" + day_dir):
                    # Extracting the documents between min_doc and max_doc.
                    if min_doc <= count < max_doc:
                        print(year_dir + "/" + month_dir + "/" + month_dir + "/" + day_dir + "/" + doc_name)
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
                        docs.append(doc)
                        summaries.append(summary)
                        print(doc)
                        print("==================")
                        print(summary)
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
def store_pas_nyt_dataset(nyt_path, min_pas, max_pas):
    docs_pas_lists = []
    refs_pas_lists = []

    docs, references = get_nyt(nyt_path, min_pas, max_pas)
    # For each document the pas_list is extracted after cleaning the text and tokenizing it.
    for doc in docs:
        print("Processing doc " + str(docs.index(doc) + min_pas) + "/400000(" + str(min_pas + max_pas) + ")")
        doc = text_cleanup(doc)
        # Splitting sentences (by dot).
        sentences = tokens(doc)
        pas_list = extract_pas(sentences, "duc")
        docs_pas_lists.append(pas_list)

    # The list of pas lists is then stored.
    with open(os.getcwd() + "/dataset/nyt_docs" + str(min_pas) + "-" + str(max_pas) + "_pas.dat", "wb") as dest_f:
        pickle.dump(docs_pas_lists, dest_f)

    # Same for reference summaries...
    for ref in references:
        print("Processing doc " + str(references.index(ref) + min_pas) + "/400000(" + str(min_pas + max_pas) + ")")
        ref = text_cleanup(ref)
        # Splitting sentences (by dot).
        sentences = tokens(ref)
        pas_list = extract_pas(sentences, "duc", keep_all=True)
        refs_pas_lists.append(pas_list)

    with open(os.getcwd() + "/dataset/nyt_refs" + str(min_pas) + "-" + str(max_pas) + "_pas.dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)
