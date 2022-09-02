import torch
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR, BATCH_SIZE, D, MODEL_NAME, CHECKPOINT_DIRNAME, \
    LEARNING_RATE, EPOCHS, CRITERION
from util.model import NN, train_model
from util.pre_proc import preproc, remove_page_lines_hp
from torch.utils.data import DataLoader
from torch import nn, optim
from util.util import absolute_path, build_data_set

device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])


def main():
    # show_model(model)
    path = absolute_path(f"/{RESOURCES_DIRNAME}/{BOOK_NAMES[CURR_BOOK_NR]}")

    remove_page_lines_hp(path)  # specific to harry potter books, they have useless lines.
    (vocabList, listOfTuples) = preproc(open(path, encoding='utf8').read(), 2)

    dataset = build_data_set(listOfTuples)  # TensorDataSet
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = NN(len(vocabList), D, vocabList).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, CRITERION, opt, dataloader, EPOCHS)

    torch.save(model, absolute_path(
        f"/{CHECKPOINT_DIRNAME}/{MODEL_NAME}"))
    # retrieve representation via "encode" function
    # TODO : separate test file in which we load the ckpt to do the tests.
    # TODO : use validation
    # TODO :  after each epoch (or in case of KeyboardInterrupt), save.


if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    # TODO : use remove_page_ligns_hp, and remove_consecutive_blank_lines
    # TODO opt: use tqdm for progress bar at each iteration of eopchs during training.
    main()
