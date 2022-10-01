import torch
from torch import optim
from tqdm import tqdm

from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, D, WORD2VEC_HOMEMADE_MODEL_NAME, \
    CHECKPOINT_DIRNAME, \
    LEARNING_RATE, EPOCHS, CRITERION, C, TRAINING_PRE_TITLE
from util.model import NN, train_model
from util.pre_proc import pre_proc, vocab_from_paths_to_text_files
from util.util import absolute_path, title_from_path, build_dl

device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])


def main():
    """
    preprocess + initialize + train
    """
    paths = list(map(lambda book_name: absolute_path(f"/{RESOURCES_DIRNAME}/{book_name}"), BOOK_NAMES))

    vocab = vocab_from_paths_to_text_files(tqdm(paths))  # make a pass through all books, establishing a vocabulary.

    model = NN(D, vocab).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for path in paths:

        (_, list_of_samples_training) = pre_proc(path, C, training=True, vocab=vocab)
        (_, list_of_samples_validation) = pre_proc(path, C, training=False, vocab=vocab)

        data_loader_training = build_dl(list_of_samples_training)
        data_loader_validation = build_dl(list_of_samples_validation)

        train_model(model, CRITERION, opt, data_loader_training, EPOCHS, len(list_of_samples_training),
                    n_validation_samples=len(list_of_samples_validation), title=TRAINING_PRE_TITLE+title_from_path(path),
                    dl_validation=data_loader_validation)

    torch.save(model, absolute_path(
        f"/{CHECKPOINT_DIRNAME}/{WORD2VEC_HOMEMADE_MODEL_NAME}"))
    # retrieve representation via "encode" function


if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    main()
