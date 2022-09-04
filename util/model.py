import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from util.constants import K

device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])


class NN(nn.Module):
    """ RNN, expects input shape (v)
    """

    def __init__(self, v: int, d: int, vocab: np.array, k=K):
        super(NN, self).__init__()
        self.k = k  # the number of candidates we want for decoding.

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

    def decode(self, vec: torch.Tensor):
        vec = vec.detach()
        distances = [torch.nn.CosineSimilarity(0)(vec, v.detach()) for v in
                     self.embeddings]  # cosine distances of each vector # cosine distances of each vector
        candidate_indices = np.argpartition(distances, -K)[-K:]  # k ***nearest*** neighbors TODO: replace with self.k
        return self.vocab[
            candidate_indices]  # word of vocab that has the closest encoding to vec according to cosine distance


def train_model(model: NN, crit, opt, dl, epochs):
    for ep in tqdm(range(epochs)):
        # Training.
        model.train()
        for it, batch in enumerate(dl):
            # 5.1 Load a batch.
            x, y = [d.to(device) for d in batch]

            # 5.2 Run forward pass.
            logits = model(x)

            # 5.3 Compute loss (using 'criterion').
            loss = crit(logits, y)

            # 5.4 Run backward pass.
            loss.backward()

            # 5.5 Update the weights using optimizer.
            opt.step()

            # 5.6 Zero-out the accumulated gradients.
            model.zero_grad()
    # TODO : use validation
    # TODO : after each epoch (or in case of KeyboardInterrupt), save.
    model.embeddings = np.array([model.encode(word).cpu() for word in model.vocab])
