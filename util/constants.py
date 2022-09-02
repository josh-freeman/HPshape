from torch import nn

BOOK_NAMES = ["HP1.txt", "HP2.txt"]
RESOURCES_DIRNAME = "examples"
GRAPH_TXT_NAME = "test.txt"
LIST_FILE_NAME_TXT = "listFile.txt"
WORD2VEC_MODEL_FILE_NAME_BIN = "gensim-model.bin"
WORD2VEC_MODEL_FILE_NAME_TXT = "gensim-model.txt"
CURR_BOOK_NR = -1
PREPROC = False
BATCH_SIZE = 128
D = 50
C= 2
MODEL_NAME = "model.pth"
CHECKPOINT_DIRNAME = "ckpt"
LEARNING_RATE = 0.01
EPOCHS = 100
CRITERION = nn.CrossEntropyLoss()
