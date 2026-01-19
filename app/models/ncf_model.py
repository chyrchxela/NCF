import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_emb = nn.Embedding(num_users + 1, emb_size)
        self.item_emb = nn.Embedding(num_items + 1, emb_size)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(emb_size + 32, 1),
            nn.Sigmoid()
        )

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)

        gmf = u * i  # GMF ветка

        mlp_in = torch.cat([u, i], dim=-1)
        mlp_out = self.mlp(mlp_in)

        x = torch.cat([gmf, mlp_out], dim=-1)
        return self.out(x).squeeze(-1)
