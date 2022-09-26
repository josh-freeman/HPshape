import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from util.constants import BATCH_SIZE


def absolute_path(relative_path):
    """
    :param relative_path: The relative path from the dir of __main__.py
    :return: absolute path from the folder containing util.
    """
    return os.path.dirname(__file__) + "/../" + relative_path


def get_total_text_from_paths(paths):
    text_final = ""
    for path in paths:
        with open(path, encoding="utf-8") as file:
            text_final += file.read()

    text_final = text_final.replace("'s", "")
    text_final = text_final.replace("'s", "")
    return text_final


def plot_losses(losses_validation: list, losses_training=None, description=""):
    x, y = zip(*enumerate(losses_validation))
    val_scatter = plt.scatter(x, y)
    x_p, y_p = zip(*enumerate(losses_training))
    tr_scatter = plt.scatter(x_p, y_p)

    plt.legend((val_scatter, tr_scatter), ("Validation", "Training"))
    plt.title(description)
    plt.savefig(description+".png")
    plt.show()


def title_from_path(path: str):
    return "" if path is None else os.path.basename(os.path.normpath(path))


def build_dl(l: list):
    return DataLoader(build_data_set(l), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def build_data_set(l: list) -> TensorDataset:
    """

        :param l: a list of (ndarray(shape=(v,1)))
        :return:
        """

    (x, y) = zip(*l)
    (tensor_x, tensor_y) = (torch.Tensor(np.array(x)), torch.Tensor(np.array(y)))

    return TensorDataset(tensor_x, tensor_y)


if __name__ == '__main__':
    pass  # normally, should not be used
