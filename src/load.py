from torch import load

from util.constants import CHECKPOINT_DIRNAME, MODEL_NAME
from util.model import NN
from util.util import absolute_path

model: NN = load(absolute_path(
    f"/{CHECKPOINT_DIRNAME}/{MODEL_NAME}"))

model.encode("harry")