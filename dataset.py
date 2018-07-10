import math
import os
import pickle
import re
import xml.etree.ElementTree
import nltk
import numpy as np

from pas import extract_pas
from utils import stem_and_stopword, text_cleanup

# Using the tokenizer to divide the text into sentences.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Compute idfs given a document list, storing them in the specified destination file.
def compute_idfs(doc_list, dest_file):
    docs_number = len(doc_list)
    # Needed to separate each sentence.
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    stems = []
    doc_stems = {}

    for doc in doc_list:
        doc_index = doc_list.index(doc)
        doc_stems[doc_index] = []
        for sent in tokenizer.tokenize(doc):
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


# Compute the PAS representation of a document and store it.
def store_pas_list(doc_name, dest_name, keep_all=False):
    with open(doc_name, "r") as text_f:
        full_text = text_cleanup(text_f.read())
        text_f.close()

    # Splitting sentences (by dot).
    sentences = tokenizer.tokenize(full_text)
    pas_list = extract_pas(sentences, "duc", keep_all=keep_all)

    with open(dest_name, "wb") as dest_f:
        pickle.dump(pas_list, dest_f)


# Load the pas list.
def load_pas_list(doc_name):
    with open(doc_name, "rb") as pas_f:
        return pickle.load(pas_f)


# Getting documents and respective summaries from DUC dataset from XML files.
def get_duc(duc_path):
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
                doc = (xml_doc.find("HEAD").text + "." + xml_doc.find("TEXT").text) if xml_doc.find("HEAD") is not None else xml_doc.find("TEXT").text

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
        sentences = tokenizer.tokenize(doc)
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
        sentences = tokenizer.tokenize(ref)
        pas_list = extract_pas(sentences, "duc", keep_all=True)
        refs_pas_lists.append(pas_list)

    with open(os.getcwd() + "/dataset/duc_refs_pas.dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)


# Getting the pas lists of documents and reference summaries.
def get_pas_lists(dataset="duc"):
    doc_path = ""
    ref_path = ""

    if dataset == "duc":
        doc_path = "/dataset/duc_docs_pas.dat"
        ref_path = "/dataset/duc_refs_pas.dat"

    with open(os.getcwd() + doc_path, "rb") as docs_f:
        docs_pas_lists = pickle.load(docs_f)
    with open(os.getcwd() + ref_path, "rb") as refs_f:
        refs_pas_lists = pickle.load(refs_f)

    return docs_pas_lists, refs_pas_lists


# Matrix representation is computed and stored.
def store_duc_matrices(dataset="duc", include_embeddings=True):
    docs_pas_lists, refs_pas_lists = get_pas_lists(dataset)

    docs_no = len(docs_pas_lists)                                   # First dimension, documents number.
    max_sent_no = max([len(doc) for doc in docs_pas_lists])         # Second dimension, max document length (sparse).
    sent_vec_len = len(docs_pas_lists[0][0].vector)                 # Third dimension, vector representation dimension.
    if include_embeddings:                                          # (w/ embeddings).
        sent_vec_len += len(docs_pas_lists[0][0].embeddings)

    # The matrix are initialized as zeros, then they'll filled in with vectors for each docs' sentence.
    docs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))
    refs_3d_matrix = np.zeros((docs_no, max_sent_no, sent_vec_len))

    for i in range(docs_no):
        for j in range(max_sent_no):
            if j < len(docs_pas_lists[i]):
                if include_embeddings:
                    docs_3d_matrix[i, j, :] = np.append(docs_pas_lists[i][j].vector, docs_pas_lists[i][j].embeddings)
                else:
                    docs_3d_matrix[i, j, :] = np.array(docs_pas_lists[i][j].vector)

            if j < len(refs_pas_lists[i]):
                if include_embeddings:
                    refs_3d_matrix[i, j, :] = np.append(refs_pas_lists[i][j].vector, refs_pas_lists[i][j].embeddings)
                else:
                    refs_3d_matrix[i, j, :] = np.array(refs_pas_lists[i][j].vector)

    # Storing the matrices in the appropriate file, depending on dataset and embedding.
    doc_path = ""
    ref_path = ""
    if dataset == "duc":
        if include_embeddings:
            doc_path = "/dataset/duc_doc_matrix.dat"
            ref_path = "/dataset/duc_ref_matrix.dat"
        else:
            doc_path = "/dataset/duc_doc_matrix_no_embed.dat"
            ref_path = "/dataset/duc_ref_matrix_no_embed.dat"

    with open(os.getcwd() + doc_path, "wb") as dest_f:
        pickle.dump(docs_3d_matrix, dest_f)
    with open(os.getcwd() + ref_path, "wb") as dest_f:
        pickle.dump(refs_3d_matrix, dest_f)


# Getting the matrices of documents and reference summaries.
def get_matrices(dataset="duc", include_embeddings=True):
    doc_path = ""
    ref_path = ""

    if dataset == "duc":
        if include_embeddings:
            doc_path = "/dataset/duc_doc_matrix.dat"
            ref_path = "/dataset/duc_ref_matrix.dat"
        else:
            doc_path = "/dataset/duc_doc_matrix_no_embed.dat"
            ref_path = "/dataset/duc_ref_matrix_no_embed.dat"

    with open(os.getcwd() + doc_path, "rb") as docs_f:
        doc_matrix = pickle.load(docs_f)
    with open(os.getcwd() + ref_path, "rb") as refs_f:
        ref_matrix = pickle.load(refs_f)

    return doc_matrix, ref_matrix
