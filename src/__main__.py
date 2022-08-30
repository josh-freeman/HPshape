from CustomIt import CustomIt
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR, PREPROC, BATCH_SIZE, D
from util.model import NN, train_model
from util.preProc import removepagelineshp, removeconsecutiveblanklines, preproc
from torch.utils.data import DataLoader
from torch import nn, optim
from util.util import absolute_path, get_doc, get_graph, print_entities_to_list_file, get_model_from_It, buildDataSet


def main():
    # show_model(model)
    with open(absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[CURR_BOOK_NR]}"), encoding="utf8") as text:
        (vocabList, listOfTuples) = preproc(text.read(), 1)

        dataset = buildDataSet(listOfTuples)  # TensorDataSet
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        model = NN(len(vocabList), D)
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.01
        epochs = 100

        opt = optim.SGD(model.parameters(), lr=learning_rate)
        train_model(model, criterion, opt, dataloader, epochs)


if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    # train?
    main()
