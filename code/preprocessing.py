import pandas as pd
import nltk
from tqdm import tqdm


# progress bar for pandas
tqdm.pandas(tqdm())

# path
path_to_data = "../data/"

# preprocessing tools
nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

# preprocessing titles
nodes_header = ["id", "year", "title", "authors", "journal", "abstract"]
nodes = pd.read_csv(path_to_data+"node_information.csv", names=nodes_header)
nodes.set_index("id", inplace=True)


# element-wise stemmed tokenization and stopwords removal for titles and abstracts
def text_element_wise_preprocess(string):
    tokens = string.lower().split(" ")
    tokens_wo_stpwds = [stemmer.stem(token) for token in tokens if token not in stpwds]
    return tokens_wo_stpwds


# element-wise lower case tokenization for authors
def authors_element_wise_preprocess(string):
    if pd.isna(string):
        return string
    tokens = string.lower().split(", ")
    return tokens


# element-wise lower case tokenization for journals
def journal_element_wise_preprocess(string):
    if pd.isna(string):
        return string
    tokens = string.lower().rstrip(".").split(".")
    return tokens

# element_wise preprocessing for journal

# apply to DF
nodes['title'] = nodes['title'].progress_apply(text_element_wise_preprocess)
nodes['abstract'] = nodes['abstract'].progress_apply(text_element_wise_preprocess)
nodes['authors'] = nodes['authors'].progress_apply(authors_element_wise_preprocess)
nodes['journal'] = nodes['journal'].progress_apply(journal_element_wise_preprocess)

# loading train
names = ["id1", "id2", "target"]
training = pd.read_csv(path_to_data + "training_set.txt", names=names, delimiter=" ")

# indexing consistent throughout project
training["my_index"] = training["id1"].astype(str) + "|" + training["id2"].astype(str)
training.set_index("my_index", inplace=True)

# same process for testing set
testing = pd.read_csv(path_to_data + "testing_set.txt", names=names, delimiter=" ")
testing["my_index"] = testing["id1"].astype(str) + "|" + testing["id2"].astype(str)
testing.set_index("my_index", inplace=True)

# save preprocessed data sets
nodes.to_csv(path_to_data + "nodes_preprocessed.csv")
training.to_csv(path_to_data + "training_new_index.txt")
testing.to_csv(path_to_data + "testing_new_index.txt")
