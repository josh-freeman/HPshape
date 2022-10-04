import numpy as np
import sklearn.cluster
import spacy
from matplotlib import pyplot as plt
from torch import load
import math
from util.constants import CHECKPOINT_DIRNAME, WORD2VEC_HOMEMADE_MODEL_NAME, K, RESOURCES_DIRNAME, BOOK_NAMES, \
    RAM_AMOUNT_SPACY_MODELS, ENTITY_LABEL_FOR_CLUSTERING
from util.model import NN, device
from util.pre_proc import remove_punctuation
from util.util import absolute_path, get_total_text_from_paths
from sklearn.cluster import KMeans


def x_is_to_y_as_blank_is_to_z(x: str, y: str, z: str, l2=True, k=K):
    x_enc = model.encode(x).to(device)
    y_enc = model.encode(y).to(device)
    z_enc = model.encode(z).to(device)
    return model.decode(x_enc - y_enc + z_enc, l2=l2, k=k)


def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = math.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = math.atan2(z, math.sqrt(XsqPlusYsq))  # theta
    az = math.atan2(y, x)  # phi
    return r, elev, az


def cart2sphA(pts):
    return np.array([cart2sph(x, y, z) for x, y, z in pts])


def people_from_paths(paths, nlp):
    nlp.max_length = RAM_AMOUNT_SPACY_MODELS
    text = get_total_text_from_paths(paths)
    text = remove_punctuation(text.strip().lower())
    doc = nlp(text[:len(text) // 50])
    return list(set([ent.text for ent in doc.ents if ent.label_ == ENTITY_LABEL_FOR_CLUSTERING]))


def asvoid(arr):
    """View the array as dtype np.void (bytes)
    This collapses ND-arrays to 1D-arrays, so you can perform 1D operations on them.
    https://stackoverflow.com/a/16216866/190597 (Jaime)"""
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def find_index(arr, x):
    arr_as1d = asvoid(arr)
    x = asvoid(x)
    return np.nonzero(arr_as1d == x)[0]


def plot_matrix(X, labels, name):
    tsvd_model = sklearn.decomposition.TruncatedSVD(n_components=7)
    X = tsvd_model.fit_transform([np.array(x) for x in X])
    tsne_model = sklearn.manifold.TSNE(n_components=3)
    Y_transform = tsne_model.fit_transform(X)
    import plotly.graph_objects as go
    Y_transform = cart2sphA(Y_transform)
    xs, ys, zs = zip(*Y_transform)
    gqsjk = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", hovertext=labels, marker=dict(
        size=12,
        color=zs,  # set color to an array/list of desired values
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    ))])
    gqsjk.show()
    gqsjk.write_html(f'../{RESOURCES_DIRNAME}/{name}.html', full_html=False, include_plotlyjs='cdn')
    return xs, ys, zs


if __name__ == '__main__':
    model: NN = load(absolute_path(
        f"/{CHECKPOINT_DIRNAME}/{WORD2VEC_HOMEMADE_MODEL_NAME}"), map_location=device)

    paths = list(map(lambda book_name: absolute_path(f"/{RESOURCES_DIRNAME}/{book_name}"), BOOK_NAMES))
    nlp = spacy.load("en_core_web_trf")
    people = people_from_paths(paths, nlp)
    #
    # # TODO split entities and flatten, and have only a unique list
    # X = np.array(list(map(lambda s: np.array(model.encode(s)), people)))
    #
    # kmeans = KMeans(n_clusters=3)
    # y_kmeans = kmeans.fit_predict(X)
    #
    # list_of_clusters = [[people[find_index(x, X)[0]] for x in X if y_kmeans[find_index(x, X)[0]] == i] for i in
    #                     range(kmeans.n_clusters)]

    # check if label of "voldemort" is the same as "tom", etc etc
    import plotly.io as pio

    pio.renderers.default = "browser"

    xs, ys, zs = plot_matrix(model.embeddings, people, "out")

    import plotly.figure_factory as ff
    import numpy as np

    group_labels = ['distplot']  # name of the dataset

    fig = ff.create_distplot([list(ys)], group_labels)
    fig.show()
