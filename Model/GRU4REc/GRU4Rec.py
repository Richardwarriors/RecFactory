import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4RecModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim = 64, hidden_size = 64,
                 num_layers = 2, padding_idx = None, loss_type = "bpr_max", neg_num = 99):
        super(GRU4RecModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.neg_num = neg_num

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size,num_layers=num_layers,
                                 batch_first=True, bias=False)
        self.fc_user = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.normalize = lambda x: F.normalize(x, p = 2, dim=-1)

    def make_padding(self, hist, padding_idx):
        return torch.where(hist == -100, hist.new_ones((1,)) * padding_idx, hist)

    def forward_user(self, user, hist):
        padding_idx = self.item_embedding.num_embeddings - 1
        hist = self.make_padding(hist, padding_idx)
        lengths = (hist != padding_idx).sum(dim=1).cpu().tolist()

        h_embedding = self.item_embedding(hist)
        packed = nn.utils.rnn.pack_padded_sequence(h_embedding, lengths, batch_first=True, enforce_sorted=False)

        #_:output sequence, h_n: hidden state
        _, h_n = self.gru(packed)

        hist_repr = h_n[-1]
        u_embedding = self.user_embedding(user)

        concat = torch.cat([u_embedding, hist_repr], dim=-1)
        u_feat = self.fc_user(concat)
        u_feat = self.normalize(u_feat)
        return u_feat

    def forward_item(self, item):
        i_feat = self.item_embedding(item)
        i_feat = self.normalize(i_feat)
        return i_feat

    def forward(self, user, pos_item, hist):
        user_feature = self.forward_user(user, hist)
        if self.loss_type == "ce":
            item_pool = torch.arange(self.item_embedding.num_embeddings - 1, devie = user_feature.device)
            item_features = self.forward_item(item_pool)
            scores = torch.matmul(user_feature, item_features.t())
            return scores
        elif self.loss_type == "bpr_max":
            pos_feature = self.forward_item(pos_item)
            neg_item = torch.randint(0, self.item_embedding.num_embeddings - 1, (pos_item.size(0), self.neg_num),device = user_feature.device)
            neg_feature = self.forward_item(neg_item)  # [B, neg_num, H]
            pos_score = torch.sum(pos_feature * user_feature, dim=-1)  # [B]
            neg_score = torch.bmm(neg_feature, user_feature.unsqueeze(-1)).squeeze(-1)  # [B, neg_num]
            return pos_score, neg_score
        else:
            raise ValueError("Unknown loss type:" + str(self.loss_type))





