import torch.nn as nn
import torch

from .transformer import TransformerBlock


class BERT(nn.Module):
    def __init__(
        self,
        sequence_len=400,
        embedding_dim=768,
        hidden_dim=768,
        output_dim=256,
        attn_heads=4,
        n_attn_layers=4,
        dropout=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(BERT, self).__init__()
        self.device = device
        self.cls_token = torch.tensor([0], device=self.device)
        self.cls = nn.Embedding(1, embedding_dim)
        self.position = nn.Embedding(sequence_len, embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, attn_heads, hidden_dim, dropout)
                for _ in range(n_attn_layers - 1)
            ]
        )
        self.transformer_blocks.insert(
            0, TransformerBlock(embedding_dim, attn_heads, hidden_dim, dropout)
        )
        self.output_dim = output_dim
        self.classifier = nn.Linear(hidden_dim, output_dim)

        self.cls.apply(weight_init)
        self.transformer_blocks.apply(weight_init)
        self.classifier.apply(weight_init)
        self.position.apply(weight_init)

    def forward(self, batch, key="items"):
        items = batch[key]
        cls_emb = self.cls(self.cls_token).unsqueeze(0).repeat(items.size()[0], 1, 1)
        embedded = torch.cat([cls_emb, items], dim=1)
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
