from CustomIt import CustomIt
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR, PREPROC
from util.preProc import removepagelineshp, removeconsecutiveblanklines
from util.util import absolute_path, get_doc, get_graph, print_entities_to_list_file, get_model_from_It


def main():
    model = get_model_from_It(CustomIt())
    # show_model(model)
    cfg = get_cfg()
    from spacy.lang.en import English
    nlp = English.from_config(cfg)
    nlp.vocab.vectors.from_disk(absolute_path("/examples/spacy/vocab"))

    with open(absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[CURR_BOOK_NR]}"), encoding="utf8") as text:
        doc = get_doc(nlp, text)
        # set similarity function of doc
        sim_graph = get_graph(doc)
        # show_graph_with_labels(sim_graph, [ent.text for ent in doc.ents[:100]])
        print_entities_to_list_file(doc, text)


def get_cfg():
    import configparser
    config = configparser.RawConfigParser()
    config.read("config.cfg")
    ret = dict()
    for section in config.sections():
        ret[section] = dict(config.items(section))
    return ret


if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    # train?
    if PREPROC:
        path = absolute_path(f"{RESOURCES_DIRNAME}/{BOOK_NAMES[CURR_BOOK_NR]}")
        removepagelineshp(path)
        removeconsecutiveblanklines(path)
    main()
