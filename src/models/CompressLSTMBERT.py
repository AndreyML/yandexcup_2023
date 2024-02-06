import torch.nn as nn
import torch

from settings import VOCAB_SIZE
from .transformer import TransformerBlock


class CompressLSTMBERT(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_dim=768,
        hidden_dim=128,
        output_dim=256,
        n_layers=4,
        attn_heads=1,
        n_attn_layers=1,
        dropout=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        activation="GELU",
    ):
        super(CompressLSTMBERT, self).__init__()
        self.device = device
        self.cls_token = torch.tensor([0], device=self.device)
        self.cls = nn.Embedding(1, embedding_dim)
        name_to_activation = {
            "SELU": nn.SELU(),
            "RELU": nn.ReLU(),
            "GELU": nn.GELU(),
            "TANH": nn.Tanh(),
        }
        activation = name_to_activation[activation]
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            activation,
            nn.Linear(embedding_dim, hidden_dim),
            activation,
        )
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, attn_heads, hidden_dim, dropout)
                for _ in range(n_attn_layers)
            ]
        )
        self.output_dim = output_dim
        self.classifier = nn.Linear(hidden_dim, output_dim)

        self.cls.apply(weight_init)
        self.lstm.apply(weight_init)
        self.transformer_blocks.apply(weight_init)
        self.classifier.apply(weight_init)

    def forward(self, batch, key="items"):
        items = batch[key]
        cls_emb = self.cls(self.cls_token).unsqueeze(0).repeat(items.size()[0], 1, 1)
        embedded = torch.cat([cls_emb, items], dim=1)
        x_t = embedded.transpose(0, 1)
        out = self.encoder(x_t.reshape(-1, embedded.size(-1)))
        out = out.reshape(x_t.shape[0], x_t.shape[1], -1)
        embedded = out.transpose(0, 1)
        output, (hidden, cell_state) = self.lstm(embedded)
        mask = batch["mask"]
        cls_mask = torch.tensor([0], device=self.device).repeat(mask.size()[0], 1)
        mask = torch.cat([cls_mask, mask], dim=1)
        mask = mask.unsqueeze(1).unsqueeze(3)
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, mask)
        logits = self.classifier(output[:, 0, :])
        return logits


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()
