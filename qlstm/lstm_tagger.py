import torch
import torch.nn as nn
from qlstm_pennylane import QLSTM


class LSTMTagger(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        num_layers,
        output_size,
        n_qubits=0,
        backend="default.qubit",
    ):
        super(LSTMTagger, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_qubits = n_qubits
        self.output_size = output_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if n_qubits > 0:
            print(f"Tagger will use Quantum LSTM running on backend {backend}")
            self.lstm = QLSTM(
                self.input_size, self.hidden_dim, n_qubits=n_qubits, backend=backend
            )
        else:
            print("Tagger will use Classical LSTM")
            self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, dtype=torch.float32
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, dtype=torch.float32
        ).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
