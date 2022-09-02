import numpy as np
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
    snape = model.encode("snape")
    severus = model.encode("severus")
    albus = model.encode("albus")
    dumbledore = model.encode("dumbledore")

    print(model.decode(model.encode("weasley") - model.encode("ron") + model.encode("harry")))
