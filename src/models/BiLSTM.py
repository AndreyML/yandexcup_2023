import torch.nn as nn

from settings import VOCAB_SIZE


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_dim=768,
        hidden_dim=64,
        output_dim=256,
        n_layers=6,
        bidirectional=True,
        dropout=0.1,
    ):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        items = batch["items"]
        output, (hidden, cell_state) = self.lstm(items)

        output = self.classifier(hidden[-1, :, :])
        return output
