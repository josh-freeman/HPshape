import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.constants import BOOK_NAMES, RESOURCES_DIRNAME, BATCH_SIZE, D, WORD2VEC_HOMEMADE_MODEL_NAME, \
    CHECKPOINT_DIRNAME, \
    LEARNING_RATE, EPOCHS, CRITERION, C
from util.model import NN, train_model
from util.pre_proc import pre_proc, vocab_from_paths_to_text_files
from util.util import absolute_path, build_data_set, title_from_path

device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])


def main():
    """
    preprocess + initialize + train
    """
    paths = list(map(lambda book_name: absolute_path(f"/{RESOURCES_DIRNAME}/{book_name}"), BOOK_NAMES))

    vocab = vocab_from_paths_to_text_files(tqdm(paths))  # make a pass through all books, establishing a vocabulary.

    path, *rest_of_paths = paths
    (_, list_of_samples_training) = pre_proc(path, C, training=True, vocab=vocab)
    (_, list_of_samples_validation) = pre_proc(path, C, training=False, vocab=vocab)
    model = NN(D, vocab).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    title = title_from_path(path)

    rest_of_paths.append(None)  # add an empty path to simulate a do-while
    for path in rest_of_paths:
        dataset_training = build_data_set(list_of_samples_training)  # TensorDataSet for training
        dataset_validation = build_data_set(list_of_samples_validation)  # TensorDataSet for validation
        data_loader_training = DataLoader(dataset_training, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        data_loader_validation = DataLoader(dataset_validation, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        train_model(model, CRITERION, opt, data_loader_training, EPOCHS, len(list_of_samples_training),
                    n_validation_samples=len(list_of_samples_validation), title=title,
                    dl_validation=data_loader_validation)
        title = title_from_path(path)
        (_, list_of_samples_training) = pre_proc(path, C, vocab)

    torch.save(model, absolute_path(
        f"/{CHECKPOINT_DIRNAME}/{WORD2VEC_HOMEMADE_MODEL_NAME}"))
    # retrieve representation via "encode" function


if __name__ == '__main__':
    # python -m spacy init vectors en ./examples/gensim-model.txt ./examples/spacy
    main()
