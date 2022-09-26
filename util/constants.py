import torch
from torch import nn

BOOK_NAMES = ["HP1.txt", "HP2.txt", "HP3.txt", "HP4_and_6.txt", "HP5.txt", "HP7.txt"]
RESOURCES_DIRNAME = "examples"
CHECKPOINT_DIRNAME = "ckpt"
GRAPH_TXT_NAME = "test.txt"
LIST_FILE_NAME_TXT = "listFile.txt"
WORD2VEC_MODEL_FILE_NAME_BIN = "gensim-model.bin"
WORD2VEC_MODEL_FILE_NAME_TXT = "gensim-model.txt"
CURR_BOOK_NR = -1
PREPROC = False
BATCH_SIZE = 128
D = 500
C = 2
WORD2VEC_HOMEMADE_MODEL_NAME = "model.pth"
LEARNING_RATE = 0.01
EPOCHS = 100
CRITERION = nn.CrossEntropyLoss()
DEVICE = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])
K: int = 10  # k for k nearest neighbours search.
MIN_WORD_THRESHOLD = 5  # minimum number of times a word has to appear to be included in the vocabulary.
F: int = 10  # fraction into which we want to divide our training vs evaluation samples (for F=10, 9/10 vs 1/10)
RAM_AMOUNT_SPACY_MODELS = 10000000  # allocate 2 MB of RAM for the lemmatizer
DROPOUT_RATE = 0.0  # it is probably a bad idea to use dropout here, as the network is tiny.
ENTITY_LABEL_FOR_CLUSTERING = 'PERSON'
