import torch
from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, CURR_BOOK_NR, BATCH_SIZE, D, MODEL_NAME, CHECKPOINT_DIRNAME, \
    LEARNING_RATE, EPOCHS, CRITERION, C
from util.model import NN, train_model
from util.pre_proc import pre_proc, remove_page_lines_hp
from torch.utils.data import DataLoader
from torch import nn, optim
from util.util import absolute_path, build_data_set

device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])


def main():
    paths = list(map(lambda book_name: absolute_path(f"/{RESOURCES_DIRNAME}/{book_name}"), BOOK_NAMES))

    path, *rest_of_paths = paths
    (vocab, list_of_samples) = pre_proc(path, C)
    model = NN(len(vocab), D, vocab).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    rest_of_paths.append(None)  # add an empty path to simulate a do-while
    for path in rest_of_paths:
        dataset = build_data_set(list_of_samples)  # TensorDataSet
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        train_model(model, CRITERION, opt, data_loader, EPOCHS)
        (_, list_of_samples) = pre_proc(path, C, vocab)

    # TODO: do this with the Vocab of the first book for all harry potter books

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
