import numpy as np
import pandas as pd
import ast
import nltk


# journal similarity feature
def compare_journals(journal1, journal2):
    if len(journal1) == 0 or len(journal2) == 0:
        return 0
    if journal1[0] == journal2[0]:
        return 1 + compare_journals(journal1[1:], journal2[1:])
    else:
        return 0


# nan-proof string converter wrapper
def lit_eval_nan_proof(string):
    if len(string) == 0:
        return np.nan
    else:
        return ast.literal_eval(string)


# element-wise stemmed tokenization and stopwords removal for titles and abstracts
def text_element_wise_preprocess(string):
    # pre-processing tools
    nltk.download('punkt')  # for tokenization
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()

    tokens = string.lower().split(" ")
    tokens_wo_stpwds = [stemmer.stem(token) for token in tokens if token not in stpwds]
    return tokens_wo_stpwds


# element-wise lower case tokenization for authors
def authors_element_wise_preprocess(string):
    if pd.isna(string):
        return string
    tokens = string.lower().split(", ")
    for i in range(len(tokens)):
        tokens[i] = tokens[i].split('(', 1)[0].strip(' ')
    return tokens


# element-wise lower case tokenization for journals
def journal_element_wise_preprocess(string):
    if pd.isna(string):
        return string
    tokens = string.lower().rstrip(".").split(".")
    return tokens
