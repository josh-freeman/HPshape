import gensim
import spacy

from CustomIt import CustomIt
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR
from util.util import absolute_path, get_doc, get_graph, print_entities_to_list_file


def main():
    nlp = spacy.load("en_core_web_lg")

    with open(absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[CURR_BOOK_NR]}"), encoding="utf8") as text:
        doc = get_doc(nlp, text)

        model = get_model()
        # show_model(model)

        # set similarity function of doc
        sim_graph = get_graph(doc)
        # show_graph_with_labels(sim_graph, [ent.text for ent in doc.ents[:100]])
        print_entities_to_list_file(doc, text)


def get_model():
    sentences = CustomIt()
    return gensim.models.Word2Vec(sentences=sentences)


if __name__ == '__main__':
    main()
