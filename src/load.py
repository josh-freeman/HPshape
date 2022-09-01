import torch.nn
from torch import load

from util.constants import CHECKPOINT_DIRNAME, MODEL_NAME
from util.model import NN
from util.util import absolute_path

model: NN = load(absolute_path(
    f"/{CHECKPOINT_DIRNAME}/{MODEL_NAME}"))

if __name__ == '__main__':
    harry = model.encode("harry")
    potter = model.encode("potter")

    print(torch.nn.CosineSimilarity(0)(harry, potter))
