import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PData.GRU4Rec.session_data import *
from PData.GRU4Rec.data import *
from Model.GRU4Rec.GRU4Rec import *


def set_seed(seed: int = 2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    user = torch.stack([b["user"] for b in batch])
    hist = torch.stack([b["hist"] for b in batch])
    pos  = torch.stack([b["pos"]  for b in batch])
    return user, hist, pos

def calculate_loss(model, outputs, pos_item = None):
    if model.loss_type == "ce":
        scores = outputs
        criterion = nn.CrossEntropyLoss()
        return criterion(scores, pos_item)
    elif model.loss_type == "bpr_max":
        pos_score, neg_socre = outputs
        diff = pos_score.unsqueeze(1) - neg_socre
        loss = -torch.log(torch.sigmoid(diff)).mean()
        return loss
    else:
        raise ValueError("Unknown loss_type: " + str(model.loss_type))

def hit_rate_at_k(ranked, gt, k):
    return 1.0 if gt in ranked[:k] else 0.0

def ndcg_at_k(ranked, gt, k):
    if gt in ranked[:k]:
        idx = ranked.index(gt)
        return 1.0 / np.log2(idx + 2)
    return 0.0

def evaluate(model, device, eval_loader, num_items, K=10):
    model.eval()
    hr_list, ndcg_list = [], []
    with torch.no_grad():
        for user, hist, pos in eval_loader:
            user = user.to(device); hist = hist.to(device); pos = pos.to(device)
            u_vec = model.forward_user(user, hist)
            i_all = torch.arange(num_items, device=device)
            i_vecs = model.forward_item(i_all)
            scores = torch.matmul(u_vec, i_vecs.t())
            _, indices = torch.topk(scores, k=K, dim=1)
            indices = indices.cpu().tolist()
            pos_list = pos.cpu().tolist()
            for ranked, gt in zip(indices, pos_list):
                hr_list.append(hit_rate_at_k(ranked, gt, K))
                ndcg_list.append(ndcg_at_k(ranked, gt, K))
    return np.mean(hr_list), np.mean(ndcg_list)

def training(model, train_loader, test_loader, device, num_items, epochs=100,
             lr = 1e-3, K=10, save_path = "./model.pth"):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_hr = 0.0
    for epoch in range(epochs):
        model.train()
        for user, hist, pos in train_loader:
            user = user.to(device); hist = hist.to(device); pos = pos.to(device)
            outputs = model(user, pos, hist)
            loss = calculate_loss(model, outputs, pos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        hr, ndcg = evaluate(model, device, test_loader, num_items, K=K)
        print(f"Epoch {epoch}: HR@{K}={hr:.4f}, NDCG@{K}={ndcg:.4f}")
        if hr > best_hr: best_hr = hr
        torch.save(model.state_dict(), save_path)
        print(f" Saved best model with HR={hr:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./OData/ml_1M/ratings.dat",
                        help="root path of GRU4Rec dataset")
    #parser.add_argument("--dataset", type=str, default="ml_1M",
    #                    help="dataset name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pad_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_Ks", type=int, nargs="+", default=10, help="@K")
    parser.add_argument("--neg_num", type=int, default=99, help="negative sampling")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    set_seed(args.seed)

    #train_df, test_df = train_test_split(args.data_path)
    train_df, test_df, num_users, num_items = load_and_split(args.data_path)
    sess = session_data(train_df, gap_hours=24, min_len=2, max_len=20, dedup=True)
    pairs = build_pairs(sess, max_hist=20)

    test_pairs = []
    for user, group in train_df.groupby("uid"):
        hist_items = group["iid"].tolist()

        tgt = test_df.loc[test_df["uid"] == user, "iid"].iloc[0]
        test_pairs.append((user, hist_items, tgt))

    train_data = SessionDataset(pairs, num_users, args.pad_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn)
    test_data = SessionDataset(test_pairs, num_items, args.pad_len)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}")

    model = GRU4RecModel(num_users, num_items, embedding_dim=args.dim, hidden_size=args.hidden_size,
                         num_layers=1, padding_idx=num_items,
                         loss_type="bpr_max", neg_num=args.neg_num)

    model.to(device)

    #multi GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    training(model, train_loader, test_loader, device, num_items, epochs=args.epochs, lr=args.lr, K=args.eval_Ks,
               save_path="gru4rec_best.pth")


if __name__ == "__main__":
    main()

