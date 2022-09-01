import torch
import torch.nn as nn


class NN(nn.Module):
    """ RNN, expects input shape (v)
    """

    def __init__(self, v: int, d: int):
        super(NN, self).__init__()

        self.V = v
        self.D = d

        self._fc1 = nn.Linear(v, d)
        self._fc2 = nn.Linear(d, v)

    def forward(self, x):
        x = self._fc1(x)  # for each layer, get with layer.state_dict()['weight'/'bias']
        x = self._fc2(x)

        return x  # will be made to be a distribution over possible words with nn.CrossEntropy()...

    def encode(self, x):
        return self._fc1(x)


def train_model(model, crit, opt, dl, epochs):
    for ep in range(epochs):
        # Training.
        model.train()
        for it, batch in enumerate(dl):
            # 5.1 Load a batch.
            x, y = batch

            # 5.2 Run forward pass.
            logits = model(x)

            # 5.3 Compute loss (using 'criterion').
            loss = crit(logits, y)

            # 5.4 Run backward pass.
            loss.backward()

            # 5.5 Update the weights using optimizer.
            opt.step()

            # 5.6 Zero-out the accumualated gradients.
            model.zero_grad()
