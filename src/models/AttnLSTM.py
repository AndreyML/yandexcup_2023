import torch.nn as nn
import torch

from settings import VOCAB_SIZE
from .transformer import TransformerBlock


class AttnLSTM52(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=15,
        n_layers=4,
        attn_heads=1,
        dropout=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(AttnLSTM52, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = TransformerBlock(
            hidden=hidden_dim,
            attn_heads=attn_heads,
            feed_forward_hidden=hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

        self.lstm.apply(weight_init)
        self.attention.apply(weight_init)
        self.classifier.apply(self.weight_init)

    def forward(self, batch):
        items = batch["items"]
        seq = add_cls(items, VOCAB_SIZE, self.device)
        embedded = self.embedding(seq)
        output, (hidden, cell_state) = self.lstm(embedded)
        mask = torch.ones(seq.size()[0], 1, seq.size()[1], seq.size()[1]).to(
            self.device
        )
        mask[:, :, -1, :] = 0
        attn_out = self.attention(output, mask)
        logits = self.classifier(attn_out[:, -1, :])
        return logits


def add_cls(items, vocab_size=VOCAB_SIZE, device='cpu'):
    cls_tensor = (
        torch.tensor([vocab_size + 1] * items.size()[0]).reshape(-1, 1).to(device)
    )
    return torch.cat([items, cls_tensor], dim=1)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
