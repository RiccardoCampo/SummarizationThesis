import math
import os
import pickle
import re
import xml.etree.ElementTree
from time import time
from pas import extract_pas
from utils import stem_and_stopword, text_cleanup, tokens, timer


def compute_idfs(doc_list, dest_path):
    """
    Compute idfs given a document list, storing them in the specified destination file.

    :param doc_list: list of documents from which terms are extracted.
    :param dest_path: path in which store the idfs file.
    """
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


def get_duc():
    """
    Getting documents and respective summaries from DUC dataset from XML files.

    :return: docs list, reference summaries list, doc names list.
    """
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


def store_pas_duc_dataset():
    """
    Load all the DUC documents and summaries, process them and store them.
    """
    docs_pas_lists = []
    refs_pas_lists = []

    docs, references, _ = get_duc()
    # For each document the pas_list is extracted after cleaning the text and tokenizing it.
    for doc in docs:
        print("Processing doc " + str(docs.index(doc)) + "/" + str(len(docs)))
        # Splitting sentences (by dot).
        sentences = tokens(doc)
        sentences = [text_cleanup(sentence) for sentence in sentences]
        pas_list = extract_pas(sentences, "duc")
        docs_pas_lists.append(pas_list)

    # The list of pas lists is then stored.
    with open(os.getcwd() + "/dataset/duc/duc_docs_pas.dat", "wb") as dest_f:
        pickle.dump(docs_pas_lists, dest_f)
        
    # Same for reference summaries...
    for ref in references:
        print("Processing doc " + str(references.index(ref)) + "/" + str(len(references)))
        # Splitting sentences (by dot).
        sentences = tokens(ref)
        sentences = [text_cleanup(sentence) for sentence in sentences]
        pas_list = extract_pas(sentences, "duc", keep_all=True)
        refs_pas_lists.append(pas_list)

    with open(os.getcwd() + "/dataset/duc/duc_refs_pas.dat", "wb") as dest_f:
        pickle.dump(refs_pas_lists, dest_f)


def get_nyt(nyt_path, min_doc=0, max_doc=100):
    """
    Retrieve New York Times corpus documents and summaries.

    :param nyt_path: path to nyt raw dataset.
    :param min_doc: start to get documents from this index.
    :param max_doc: stop to get documents at this index.
    :return: docs list, reference summaries list.
    """
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
                        # Adding the headline as first sentence (there is a typo in the xml).
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


def store_pas_nyt_dataset(nyt_path, min_pas, max_pas):
    """
    Load NYT documents and summaries, process them and store them.
    Process a number of documents between min_pas and max_pas.

    :param nyt_path: path to nyt raw dataset.
    :param min_pas: first document number.
    :param max_pas: last document number
    """
    docs_pas_lists = []
    refs_pas_lists = []

    docs, references = get_nyt(nyt_path, min_pas, max_pas)

    for i in range(len(docs)):
        start_time = time()
        print("Processing doc " + str(i) + "/" + str(len(docs)))
        doc = docs[i]
        ref = references[i]

        # Splitting sentences (by dot).
        sentences = tokens(doc)
        sentences = [text_cleanup(sentence) for sentence in sentences]
        doc_pas_list = extract_pas(sentences, "nyt")

        # Splitting sentences (by dot).
        sentences = tokens(ref)
        sentences = [text_cleanup(sentence) for sentence in sentences]
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


def arrange_nyt_pas_lists(dim=1000, max_len=300, max_file=660000):
    """
    Putting pas lists in files of fixed length and without excessively long documents.

    :param dim: number of docs in a file.
    :param max_len: maximum document length.
    :param max_file: last document index.
    """
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


def get_pas_lists(index=-1):
    """
    Getting the pas lists of documents and reference summaries.

    :param index: represent the batch of compact nyt pas to get or, if -1, it tells to get duc pas lists.
    :return: document and summary pas lists.
    """
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
