import nltk
import pandas as pd
from tqdm import tqdm

from feature_engineering.tools import\
    text_element_wise_preprocess,\
    authors_element_wise_preprocess,\
    journal_element_wise_preprocess

# This script reads the data in node_information.csv and training_set and testing_set.csv, and creates the
# files "nodes_preprocessed.csv", "training_new_index.txt" and "testing_new_index.txt".

# progress bar for pandas
tqdm.pandas(tqdm())

# path
path_to_data = "data/"

# pre-processing tools
nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

# pre-processing titles
nodes_header = ["id", "year", "title", "authors", "journal", "abstract"]
nodes = pd.read_csv(path_to_data+"node_information.csv", names=nodes_header)
nodes.set_index("id", inplace=True)

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
names = ["id1", "id2"]
testing = pd.read_csv(path_to_data + "testing_set.txt", names=names, delimiter=" ")
testing["my_index"] = testing["id1"].astype(str) + "|" + testing["id2"].astype(str)
testing.set_index("my_index", inplace=True)

# save preprocessed data sets
nodes.to_csv(path_to_data + "nodes_preprocessed.csv")
training.to_csv(path_to_data + "training_new_index.txt")
testing.to_csv(path_to_data + "testing_new_index.txt")
