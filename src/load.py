from torch import load

from util.constants import CHECKPOINT_DIRNAME, WORD2VEC_HOMEMADE_MODEL_NAME, K
from util.model import NN, device
from util.util import absolute_path


def x_is_to_y_as_blank_is_to_z(x: str, y: str, z: str, l2=True):
    assert all(w in model.vocab for w in (x, y, z))
    x_enc = model.encode(x).to(device)
    y_enc = model.encode(y).to(device)
    z_enc = model.encode(z).to(device)
    return model.decode(x_enc - y_enc + z_enc, l2=l2)


if __name__ == '__main__':
    model: NN = load(absolute_path(
        f"/{CHECKPOINT_DIRNAME}/{WORD2VEC_HOMEMADE_MODEL_NAME}"), map_location=device)
    model.k = K
    print(model.decode(model.encode("harry"), k=10, l2=True))
    print(model.decode(model.encode("harry"), k=10, l2=False))
