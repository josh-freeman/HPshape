from torch import load

from util.constants import CHECKPOINT_DIRNAME, MODEL_NAME
from util.model import NN, device
from util.util import absolute_path


def x_is_to_y_as_blank_is_to_z(x, y, z):
    x_enc = model.encode(x).to(device)
    y_enc = model.encode(y).to(device)
    z_enc = model.encode(z).to(device)
    return model.decode(x_enc - y_enc + z_enc)


if __name__ == '__main__':
    model: NN = load(absolute_path(
        f"/{CHECKPOINT_DIRNAME}/{MODEL_NAME}"), map_location=device)

    print(x_is_to_y_as_blank_is_to_z("wizard", "warlock", "harry"))
    print(x_is_to_y_as_blank_is_to_z("harry", "potter", "dursley"))
