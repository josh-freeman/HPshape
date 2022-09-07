import os
import re
from os.path import exists
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import gensim
import spacy
from unidecode import unidecode as decode

from util.constants import GRAPH_TXT_NAME, RESOURCES_DIRNAME, LIST_FILE_NAME_TXT, WORD2VEC_MODEL_FILE_NAME_BIN, \
    WORD2VEC_MODEL_FILE_NAME_TXT, BATCH_SIZE


def absolute_path(relative_path):
    """
    :param relative_path: The relative path from the dir of __main__.py
    :return: absolute path from the folder containing util.
    """
    return os.path.dirname(__file__) + "/../" + relative_path


def normalize(s):
    """
    normalize a single string
    :param s:the string to normalize
    :return:normalized string (lowered and unidecode.decoded)
    """
    return decode(s).lower().strip()


def distinct(lst):
    """
    remove duplicates from list
    :param lst: list to remove duplicates from
    :return: list free of duplicates
    """
    return list(dict.fromkeys(lst))


import matplotlib.pyplot as plt
import networkx as nx


def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()


def get_doc(nlp, text):
    entireBook = text.read()

    nlp.add_pipe("ner")
    ruler = nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": True})

    nlp.add_pipe('lemmatizer', before='ner', config={"mode": "lookup"})
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("parser")  # for nouns
    nlp.add_pipe("doc_cleaner")
    nlp.initialize()

    patterns = [{"label": "PER", "pattern": [{"LOWER": "harry"}, {"LOWER": "potter"}]},
                {"label": "PER", "pattern": [{"LOWER": "severus"}, {"LOWER": "snape"}]},
                {"label": "PER", "pattern": [{"LOWER": "ronald"}, {"LOWER": "weasley"}]},
                {"label": "PER", "pattern": [{"LOWER": "albus"}, {"LOWER": "percival"}, {"LOWER": "dumbledore"}]},
                ]

    specific_forms_patterns_persons = [
        {"label": "PER", "pattern": [{"ENT_TYPE": "PER"}, {"ORTH": ","}, {"ENT_TYPE": "PER"}]},
        {"label": "PER", "pattern": [{"ENT_TYPE": "PER"}]},
        {"label": "PER", "pattern": [{"ENT_TYPE": "PER"}, {"OP": "+"}]},
    ]
    ruler.add_patterns(patterns)
    ruler.add_patterns(specific_forms_patterns_persons)
    doc = nlp(entireBook[100:1000000])
    return doc


def get_model_from_it(customIt):
    filepath_bin = absolute_path(f"/{RESOURCES_DIRNAME}/{WORD2VEC_MODEL_FILE_NAME_BIN}")
    filepath_txt = absolute_path(f"/{RESOURCES_DIRNAME}/{WORD2VEC_MODEL_FILE_NAME_TXT}")
    if not exists(filepath_bin):
        model = gensim.models.Word2Vec(sentences=customIt)
        model.save(filepath_bin)
        model.wv.save_word2vec_format(filepath_txt)
    else:
        model = gensim.models.Word2Vec.load(filepath_bin)
    return model


def get_graph(doc):
    filepath = absolute_path(f"/{RESOURCES_DIRNAME}/{GRAPH_TXT_NAME}")
    if not exists(filepath):
        n = len(doc.ents[:100])
        return graphFromDoc(doc, n)
    else:
        return np.recfromcsv(filepath)


def graphFromDoc(doc, n):
    simGraph = np.zeros((n, n))
    for i in range(len(doc.ents[:100])):
        entI = doc.ents[i]
        # print(entI.text, entI.start_char, entI.end_char, entI.label_,file=file)
        for j in range(len(doc.ents[:100])):
            entJ = doc.ents[j]
            simGraph[i, j] = entI.similarity(entJ)
    np.savetxt("adj.csv", simGraph, delimiter=",")
    return simGraph


def print_entities_to_list_file(doc, text):
    with open(absolute_path(f"/{RESOURCES_DIRNAME}/{LIST_FILE_NAME_TXT}"), "w", encoding="utf8") as listFile:
        for ent in doc.ents:
            print(ent.text, ent.label_, ent.start, file=listFile)
        text.close()


##THE FOLLOWING FUNCTIONS ARE FROM https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
from sklearn.manifold import TSNE  # final reduction
import numpy as np  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    return plt


##END OF GIVEN FUNCTIONS
def plot_losses(losses_validation: list, losses_training=None, description=""):
    x, y = zip(*enumerate(losses_validation))
    val_scatter = plt.scatter(x, y)
    x_p, y_p = zip(*enumerate(losses_training))
    tr_scatter = plt.scatter(x_p, y_p)

    plt.legend((val_scatter, tr_scatter), ("Validation", "Training"))
    plt.title(description)
    plt.show()


def title_from_path(path: str):
    return "" if path is None else os.path.basename(os.path.normpath(path))


def show_model(model):
    x_vals, y_vals, labels = reduce_dimensions(model)
    plot_with_matplotlib(x_vals, y_vals, labels).show()


if __name__ == '__main__':
    pass  # normally, should not be used


def build_dl(l:list):
    return DataLoader(build_data_set(l), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def build_data_set(l: list) -> TensorDataset:
    """

    :param l: a list of (ndarray(shape=(v,1)))
    :return:
    """

    (x, y) = zip(*l)
    (tensor_x, tensor_y) = (torch.Tensor(np.array(x)), torch.Tensor(np.array(y)))

    return TensorDataset(tensor_x, tensor_y)
