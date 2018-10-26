import os
import numpy as np

from nltk.stem.porter import *
from pntl.tools import Annotator
from utils import stem_and_stopword, tf_idf, remove_punct, sentence_embeddings, centrality_scores

if os.name == "posix":
    SENNA_PATH = "/home/arcslab/Documents/Riccardo_Campo/tools/senna"
    STANFORD_PATH = "/home/arcslab/Documents/Riccardo_Campo/tools/stanford-parser"
else:
    SENNA_PATH = "C:/Users/Riccardo/Documents/senna"
    STANFORD_PATH = "C:/Users/Riccardo/Documents/stanford-parser"

# Initializing the annotator with the specified paths.
annotator = Annotator(senna_dir=SENNA_PATH, stp_dir=STANFORD_PATH)


# This class contains all the necessary information about a PAS.
class Pas:
    def __init__(self, sentence, parts_of_speech, position, pas_no, raw_pas, out_of_order):
        # Reference sentence.
        self.sentence = sentence
        # POS tags.
        self.parts_of_speech = parts_of_speech
        # Raw SENNA output.
        self.raw_pas = raw_pas
        # Reference sentence position.
        self.position = position
        # Number of the PAS (referring to the PASs extracted from the reference sentence).
        self.pas_no = pas_no
        # Realized PAS.
        self.realized_pas = ""
        # Tells if the realized pas doesn't respect the order of the original sentence.
        self.out_of_order = out_of_order
        # PAS embeddings.
        self.embeddings = []
        # PAS vectorial representation.
        # Position score, reference sentence length score, tf_idf, numerical data, centrality, title.
        self.vector = [0, 0, 0, 0, 0, 0]

    # Define equality as equality between vector representations.
    def __eq__(self, other):
        return self.vector == other.vector

    def __str__(self):
        return "\nPas representation: " + str(self.vector) + "\npas number: " + str(self.pas_no) + "\n" + \
               "full sentence: \n" + self.sentence + "\n" + "raw pas: \n" + str(self.raw_pas) + "\n" + \
               "pos: " + "\n" + str(self.parts_of_speech) + "\n"

    # Completing the PAS after the initialization with pas realization and embeddings.
    def complete_pas(self,
                     realized_pas,
                     embeddings,
                     sentence_no,
                     longest_sent_len,
                     tf_idfs,
                     pas_centrality,
                     title_similarity):
        self.embeddings = embeddings
        self.realized_pas = realized_pas
        position_score = (sentence_no - self.position) / sentence_no

        tf_idf_score = 0
        numerical_score = 0

        terms = list(set(stem_and_stopword(" ".join(x for x in self.raw_pas.values()))))
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

        self.vector = [position_score, len(self.sentence) / longest_sent_len, tf_idf_score,
                       numerical_score / len(self.sentence), pas_centrality, title_similarity]


# Produce the sentence realization given a PAS
def realize_pas(pas):
    phrase = ""
    raw_pas = pas.raw_pas

    # Adding spaces to avoid errors like finding "he" in "the"
    # Removing punctuation, just need to find the position of the arguments.
    full_sent = remove_punct(" " + pas.sentence + " ")
    args_positions = []

    # For every value of the SRL dictionary, the position of this value is found in the original sentence and
    # placed in a list which is sorted by position afterwards.
    redundant_modals = ["going", "has", "have", "had"]
    for arg_key in raw_pas.keys():
        arg_val = raw_pas[arg_key]

        # Excluding "not" and modals that might be repeated in the verb fixing process.
        if arg_key != "AM-NEG" and not (arg_key == "AM-MOD" and arg_val in redundant_modals):
            arg = (" " + remove_punct(arg_val) + " ").replace("  ", " ")
            arg_index = full_sent.find(arg)
            # Verbs has to be fixed as SENNA clears auxiliaries and modals.
            if arg_key == "V":
                arg_val = fix_verb(pas)
            arg_pos = (arg_index, arg_val)
            args_positions.append(arg_pos)

    # Sorting the arguments
    sorted_args = sorted(args_positions, key=lambda tup: tup[0])

    # Building the phrase by spacing the arguments.
    for arg_pos in sorted_args:
        phrase += arg_pos[1] + " "

    # De-spacing the contracted form (I 'm to I'm).
    phrase = re.sub("([a-zA-Z0-9]) \'([a-zA-Z0-9])", r"\1'\2", phrase)
    # De-spacing apices and parentheses (' " " '  to  '" "').
    phrase = re.sub("\" ([a-zA-Z0-9 ,']+) \"", r'"\1"', phrase)
    phrase = re.sub("\( ([a-zA-Z0-9 ,']+) \)", r'"\1"', phrase)
    # De-spacing punctuation.
    phrase = re.sub(" [.,:;] ", r", ", phrase)

    return phrase


# Fixes the verb by checking on previous verbs/auxiliaries in the original sentence.
def fix_verb(pas):
    raw_pas = pas.raw_pas
    pos = dict(pas.parts_of_speech)
    verb = raw_pas["V"]

    words = remove_punct(pas.sentence).split()
    verb_index = 0
    # Fetching the verb location in the original sentence.
    if verb in words:
        verb_index = words.index(verb)

    # Checking if in 4 words preceding the verb are auxiliaries/modals.
    if verb in pos.keys():
        if pos[verb].startswith("VB"):
            verb_prefix = ""
            for i in range(1, 5):
                if check_prev_verb(words, pos, verb_index - i):
                    verb_prefix = words[verb_index - i] + " " + verb_prefix
                else:
                    break
            # Excluding the cases in which the only part added is "to".
            if not (verb_prefix.startswith("to")):
                verb = verb_prefix + verb
    return verb


# Check if the previous verb is a verb or "not" or "to".
def check_prev_verb(words, pos, index):
    if index >= 0:
        if words[index] in pos.keys():
            if pos[words[index]].startswith("VB") or words[index] == "not" or words[index] == "to":
                return True
    return False


# Extracts the PASs from a list of sentences (dataset name is needed to fetch the proper IDF file).
def extract_pas(sentences, dataset_name, keep_all=False):
    # Compute the TFIDF vector of all terms in the document.
    tf_idfs = tf_idf(sentences, os.getcwd() + "/dataset/" + dataset_name + "/" + dataset_name + "_idfs.dat")

    # Longest sentence length needed afterwards for the length score.
    longest_sent_len = max(len(sent) for sent in sentences)

    pas_list = []
    for sent in sentences:
        # Ignoring short sentences (errors).
        print(len(sent))
        if 3 < len(remove_punct(sent)) and len(sent) < 1000:
            sent_index = sentences.index(sent)

            # Substituting single apices with double apices to avoid errors with SRL.
            sent = re.sub("\'([a-zA-Z0-9])([a-zA-Z0-9 ]+)([a-zA-Z0-9])\'", r'" \1\2\3 "', sent)
            print(sent)

            annotations = annotator.get_annoations(remove_punct(sent).split())
            # Getting SRL annotations from SENNA.
            sent_srl = annotations['srl']
            # Getting POS tags from SENNA.
            parts_of_speech = annotations['pos']

            for raw_pas in sent_srl:
                accept_pas = 1
                out_of_order = 0
                chk_sent = remove_punct(sent)
                # Rejecting PASs with arguments that change the order (w.r.t. to the one of the original sentence);
                # These represents the 10% of the total PASs and the 80% of them are incorrect.
                for arg in raw_pas.values():
                    # Replacing double spaces with a single space to avoid some arguments to be ignored.
                    arg = remove_punct(arg.replace("  ", " "))

                    if chk_sent.find(arg) < 0:
                        accept_pas = keep_all                   # This will reject the pas if keep_all = False.
                        out_of_order = 1
                        break

                if accept_pas:
                    pas = Pas(sent,
                              parts_of_speech,
                              sent_index,
                              sent_srl.index(raw_pas),
                              raw_pas,
                              out_of_order)
                    pas_list.append(pas)

    # Completing each PAS with its realization embeddings and vector representation.
    # This process is done after the initialization as all the other pas are needed.
    realized_pass = []
    for pas in pas_list:
        realized_pass.append(realize_pas(pas))

    # Here the title is put together with the pass to avoid starting another embedding process
    realized_pass.append(sentences[0])
    pas_embeddings = sentence_embeddings(realized_pass)

    # Get the centrality scores for the pas embeddings
    pas_centralities = centrality_scores(pas_embeddings)

    for pas in pas_list:
        pas_index = pas_list.index(pas)
        pas.complete_pas(realized_pass[pas_index],
                         pas_embeddings[pas_index],
                         len(sentences),
                         longest_sent_len,
                         tf_idfs,
                         pas_centralities[pas_index],
                         np.inner(np.array(pas_embeddings[pas_index]), np.array(pas_embeddings[-1])))

    return pas_list
