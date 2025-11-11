import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRMF(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim = 64, reg=1e-4, use_bias=False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.reg = reg
        self.use_bias = use_bias

        self.embedding_user = nn.Embedding(num_users, embedding_dim)
        self.embedding_item = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)

        if use_bias:
            self.bu = nn.Embedding(num_users,1)
            self.bi = nn.Embedding(num_items,1)
            nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)

    def forward(self, users, items):
        u = self.embedding_user(users)
        i = self.embedding_item(items)
        s = (u * i).sum(dim=-1)
        if self.use_bias:
            s = s + self.bu(users).squeeze(-1) + self.bi(items).squeeze(-1)
        return s

    def bpr_loss(self, users, pos_items, neg_items):

        y_ui = self.forward(users, pos_items)
        y_uj = self.forward(users, neg_items)

        bpr = -F.logsigmoid(y_ui - y_uj).mean()

        #l2 regularization
        u = self.embedding_user(users)
        i_pos = self.embedding_item(pos_items)
        i_neg = self.embedding_item(neg_items)
        l2 = (u.pow(2).sum() + i_pos.pow(2).sum() + i_neg.pow(2).sum()) / users.size(0)

        if self.use_bias:
            l2 = l2 + 1e-6 * (
                self.bu(users).pow(2).sum()
                + self.bi(pos_items).pow(2).sum()
                + self.bi(neg_items).pow(2).sum()
            ) / users.size(0)

        return bpr + self.reg * l2
