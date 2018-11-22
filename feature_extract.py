import spacy
import lyx
from enum import Enum
import collections
from spacy.attrs import HEAD
import numpy
from benepar.spacy_plugin import BeneparComponent
POS = Enum('POS',  ("Tag", "-LRB-", "-RRB-", ",", ":", ".", "''", "\"\"", "#", "``", "$", "ADD", "AFX", "BES", "CC", "CD", "DT", "EX", "FW", "GW", "HVS", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD",
                    "NFP", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "_SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX"))

ENTITY = Enum('ENTITY',  ('PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                          'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'))

DEP = Enum('DEP', ("acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp")
           )

en_freq = lyx.io.load_pkl("en_freq")
en_freq = collections.defaultdict(lambda: 0, en_freq)
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent("benepar_en"))


class Features():
    def __init__(self, text):
        self.text = text
        self.rank = en_freq[text]
        pass


def find_root(token, depth):
    if token.dep_ == 'ROOT':
        return depth
    parent = token.ancestors.next
    find_root(parent)


def feature_extract(sent):
    sent = sent.replace(")", "").replace("(", "(")
    doc = nlp(sent)
    features = []
    sent = list(doc.sents)[0]
    tree_str = sent._.parse_string
    deps = []
    depth = 0

    for i, c in enumerate(tree_str):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if tree_str[i-1] != ")":
                deps.append(depth)

    for token in doc:
        feature = Features(token.text)
        feature.lemma = token.lemma_
        feature.tag = token.tag_
        feature.dep = token.dep_
        feature.shape = token.shape_
        feature.is_alpha = token.is_alpha
        feature.is_digit = token.is_digit
        feature.is_title = token.is_title
        feature.like_num = token.like_num
        feature.is_lower = token.is_lower
        feature.is_upper = token.is_upper
        feature.is_currency = token.is_currency
        feature.is_punct = token.is_punct
        feature.is_stop = token.is_stop
        feature.is_oov = token.is_oov
        feature.vector_norm = token.vector_norm
        feature.con_dep = token.con_dep
        feature.offset = abs(token.head.i-token.i)
        feature.dep_offset = abs(token.head.i-token.i)
        features.append(feature)

    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            features[i].ner = ent.label

    noun_chunks = list(doc.noun_chunks)
    for chunk in noun_chunks:
        for i in range(chunk.start, chunk.end):
            feature.ischunk = 1

    return features


def main():

    sentences = lyx.io.read_all_lines("input.txt")
    sentFeature = list(map(feature_extract, sentences))
    lyx.io.save_pkl(sentFeature, "sentFeature")


if __name__ == "__main__":
    main()
