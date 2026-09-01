"""
Architecture du LSTM Autoencodeur — DOIT rester identique à celle utilisée
lors de l'entraînement dans Colab, sinon torch.load_state_dict échoue.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=64, latent=16):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.to_latent = nn.Linear(hidden, latent)
        self.to_hidden = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        seq_len = x.shape[1]
        _, (h, _) = self.enc(x)
        z = self.to_latent(h[-1])
        h0 = self.to_hidden(z).unsqueeze(1).repeat(1, seq_len, 1)
        y, _ = self.dec(h0)
        return self.out(y)
