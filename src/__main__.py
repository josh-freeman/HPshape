import torch

from custom_it import CustomIt
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR, BATCH_SIZE, D
from util.model import NN, train_model
from util.pre_proc import preproc
from torch.utils.data import DataLoader
from torch import nn, optim
from util.util import absolute_path, build_data_set


def main():
    # show_model(model)
    with open(absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[CURR_BOOK_NR]}"), encoding="utf8") as text:
        (vocabList, listOfTuples) = preproc(text.read(), 1)

        dataset = build_data_set(listOfTuples)  # TensorDataSet
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        model = NN(len(vocabList), D)
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.01
        epochs = 100

        opt = optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, criterion, opt, dataloader, epochs)

        torch.save(model,)# TODO: retrieve representation(s?) from the model's linear layers


if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    # train?
    main()
