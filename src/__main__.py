from util.constants import BOOK_NAMES, RESOURCES_DIRNAME
from util.util import absolute_path


def main():
    import spacy

    # Requirements: pip install spacy-lookups-data
    nlp = spacy.load("en_core_web_sm")
    nlp.remove_pipe("lemmatizer")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()

    with open(absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[-1]}"), encoding="utf8") as text:
        entireBook = text.read()
        doc = nlp(entireBook[100:100000])
        nlp.add_pipe("entity_ruler", before="ner")
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
        text.close()


if __name__ == '__main__':
    main()

