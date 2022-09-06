import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from util.constants import K
from util.util import plot_losses

device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])


class NN(nn.Module):
    """ RNN, expects input shape (v)
    """

    def __init__(self, d: int, vocab: np.array, k=K):
        super(NN, self).__init__()
        self.k = k  # the number of candidates we want for decoding.

        v = len(vocab)
        self.vocab = vocab
        self.embeddings = None  # to be updated after training

        self._fc1 = nn.Linear(v, d)
        self._fc2 = nn.Linear(d, v)

    def forward(self, x):
        x = self._fc1(x)  # for each layer, get with layer.state_dict()['weight'/'bias']
        x = self._fc2(x)

        return x  # will be made to be a distribution over possible words with nn.CrossEntropy()...

    def encode(self, word: str) -> torch.Tensor:
        return self._fc1(torch.Tensor(np.where(self.vocab == word.lower(), 1, 0)).to(device)).detach()

    def decode(self, vec: torch.Tensor, k=None, l2=True):
        def metric(x):
            if l2:
                return (vec - x.detach()).pow(2).sum()
            else:
                return -torch.nn.CosineSimilarity(0)(vec, x)

        k = self.k if k is None else k
        vec = vec.detach()
        distances = [metric(v) for v in
                     self.embeddings]  # cosine distances of each vector # cosine distances of each vector
        candidate_indices = np.argpartition(distances, k)[:k]  # k ***nearest*** neighbors
        return self.vocab[
            candidate_indices]  # word of vocab that has the closest encoding to vec according to cosine distance


def train_model(model: NN, crit, opt, dl_train, epochs, n_train_samples: int, n_validation_samples=None,
                title="", dl_validation=None):
    average_validation_losses = []
    average_training_losses = []
    for ep in tqdm(range(epochs)):
        total_loss = 0
        # Training.
        model.train()
        for it, batch in enumerate(dl_train):
            # 5.1 Load a batch.
            x, y = [d.to(device) for d in batch]

            # 5.2 Run forward pass.
            logits = model(x)

            # 5.3 Compute loss (using 'criterion').
            loss = crit(logits, y)

            total_loss += loss

            # 5.4 Run backward pass.
            loss.backward()

            # 5.5 Update the weights using optimizer.
            opt.step()

            # 5.6 Zero-out the accumulated gradients.
            model.zero_grad()
        average_training_loss = total_loss / n_train_samples
        average_training_losses.append(average_training_loss.item())
        if dl_validation is not None and n_validation_samples is not None:
            model.eval()
            with torch.no_grad():
                sum_of_validation_losses = 0
                for iteration, batch in enumerate(dl_validation):
                    # Get batch of data.
                    x, y = [d.to(device) for d in batch]
                    out_data = model(x)
                    sum_of_validation_losses += crit(out_data, y)
                average_validation_loss = sum_of_validation_losses / n_validation_samples
                average_validation_losses.append(average_validation_loss.item())
    plot_losses(average_validation_losses, losses_training=average_training_losses,
                description=f"Loss as a function of epoch for {title}")
    model.embeddings = np.array([model.encode(word).cpu() for word in model.vocab])
