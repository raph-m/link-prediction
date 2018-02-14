import ast
import numpy as np


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




def get_tf_idf_encoding(index):
    tfidf = models.TfidfModel(dictionary=dictionary)
    abstract = nodes.at[index, "abstract"]
    abstract = abstract.split(" ")
    abstract = dictionary.doc2bow(abstract)
    ans = tfidf[[abstract]]
    return ans[0]