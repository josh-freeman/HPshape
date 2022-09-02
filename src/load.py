from torch import load

from util.constants import CHECKPOINT_DIRNAME, MODEL_NAME
from util.model import NN, device
from util.util import absolute_path


if __name__ == '__main__':
    model: NN = load(absolute_path(
        f"/{CHECKPOINT_DIRNAME}/{MODEL_NAME}"),map_location=device)

    harry = model.encode("harry").to(device)
    potter = model.encode("potter").to(device)
    snape = model.encode("snape").to(device)
    severus = model.encode("severus").to(device)
    albus = model.encode("albus").to(device)
    dumbledore = model.encode("dumbledore").to(device)
    weasley = model.encode("weasley").to(device)

    test1 = model.encode("lily").to(device)
    dudley = model.encode("dudley").to(device)
    poudlard = model.encode("hogwarts").to(device)
    magic = model.encode("magic").to(device)

    print((harry-test1).pow(2).sum().pow(1/2))
    print(dudley.pow(2).sum().pow(1/2))
    print(model.decode(harry - test1 + dudley))
    print(model.decode(poudlard - magic))
