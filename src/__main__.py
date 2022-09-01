import torch

from custom_it import CustomIt
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR, BATCH_SIZE, D, MODEL_NAME, CHECKPOINT_DIRNAME, \
    LEARNING_RATE, EPOCHS
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

        model = NN(len(vocabList), D, vocabList)
        criterion = nn.CrossEntropyLoss()

        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(model, criterion, opt, dataloader, EPOCHS)

        torch.save(model, absolute_path(
            f"/{CHECKPOINT_DIRNAME}/{MODEL_NAME}"))
        # retrieve representation via "encode" function
        # TODO : separate test file in which we load the ckpt to do the tests.
        # TODO : use validation
        # TODO :  after each epoch (or in case of KeyboardInterrupt), save.


if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    # train?
    main()
