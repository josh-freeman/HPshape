import gensim
import spacy

from CustomIt import CustomIt
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR
from util.util import absolute_path, get_doc, get_graph, print_entities_to_list_file, get_model_from_It, show_model


def main():
    model = get_model_from_It(CustomIt())
    # show_model(model)
    nlp = spacy.Language.from_config(???)

    with open(absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[CURR_BOOK_NR]}"), encoding="utf8") as text:
        doc = get_doc(nlp, text)
        # set similarity function of doc
        sim_graph = get_graph(doc)
        # show_graph_with_labels(sim_graph, [ent.text for ent in doc.ents[:100]])
        print_entities_to_list_file(doc, text)




if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    # train?
    main()
