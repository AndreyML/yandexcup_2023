import torch.nn as nn
import torch

from .transformer import TransformerBlock


class ConvBERT52(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=256,
        output_dim=256,
        attn_heads=4,
        n_attn_layers=4,
        dropout=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(ConvBERT52, self).__init__()
        self.device = device
        self.cls_token = torch.tensor([0], device=self.device)
        self.cls = nn.Embedding(1, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=2 * hidden_dim,
                kernel_size=5,
                stride=1,
                padding=4,
                dilation=2,
            ),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=2 * hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
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
        self.transformer_blocks.apply(weight_init)
        self.classifier.apply(weight_init)

    def forward(self, batch, key="items"):
        items = batch[key]
        cls_emb = self.cls(self.cls_token).unsqueeze(0).repeat(items.size()[0], 1, 1)
        embedded = torch.cat([cls_emb, items], dim=1)
        embedded = self.encoder(embedded.transpose(1, 2)).transpose(1, 2)
        mask = batch["mask"]
        cls_mask = torch.tensor([0], device=self.device).repeat(mask.size()[0], 1)
        mask = torch.cat([cls_mask, mask], dim=1)
        mask = mask.unsqueeze(1).unsqueeze(3)
        for transformer in self.transformer_blocks:
            output = transformer.forward(embedded, mask)
        logits = self.classifier(output[:, 0, :])
        return logits


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()
