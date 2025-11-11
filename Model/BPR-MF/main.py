import argparse
import pandas as pd
import numpy as np
import torch

from BPR_MF import BPRMF
from PData.BPR_MF.data import DataLoader

def set_seed(seed: int = 2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../../PData/BPR_MF",
                        help="root path of BPR_MF dataset")
    parser.add_argument("--dataset", type=str, default="ml_1M",
                        help="dataset name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4, help="l2 regularization")
    parser.add_argument("--use_bias", action="store_true", help="user or item bias")
    parser.add_argument("--eval_Ks", type=int, nargs="+", default=[10], help="@K")
    parser.add_argument("--eval_n_neg", type=int, default=99, help="negative sampling")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    #parser.add_argument("--grad_clip", type=float, default=5.0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}")

    dl = DataLoader(root_path=args.root_path, dataset=args.dataset)
    n_items = dl.max_item
    n_users = dl.n_user
    print(f"number of users: {n_users}, number of items: {n_items}, number of interactions: {dl.cnt}")

    #Model
    model = BPRMF(n_users, n_items, embedding_dim=args.dim, reg=args.reg, use_bias=args.use_bias)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.eval()
    with torch.no_grad():
        metrics = dl.evaluate(model, device, Ks=tuple(args.eval_Ks), n_neg=args.eval_n_neg)
        print(f"[Epoch 0] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0, 0
        for users, pos_items, neg_items in dl.generate_data(batch_size=256):
            if len(users) ==0: continue

            users = torch.tensor(users, dtype = torch.long, device = device)
            pos = torch.tensor(pos_items, dtype = torch.long, device = device)
            neg = torch.tensor(neg_items, dtype = torch.long, device = device)

            opt.zero_grad()
            loss = model.bpr_loss(users, pos, neg)
            loss.backward()

            #if args.grad_clip and args.grad_clip > 0:
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            opt.step()

            total_loss += float(loss.detach().cpu().item())
            steps += 1

        avg_loss = total_loss / max(1, steps)

        # === Evaluation phase ===
        model.eval()
        with torch.no_grad():

            metrics = dl.evaluate(model, device, Ks=tuple(args.eval_Ks), n_neg=args.eval_n_neg)

        # === Print results for this epoch ===
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[Epoch {ep:03d}] loss={avg_loss:.4f}  {metric_str}")
    model.eval()
    with torch.no_grad():
        try:
            metrics = dl.evaluate(model, device, Ks=tuple(args.eval_Ks), n_neg=args.eval_n_neg)
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[Epoch {ep:03d}] loss={avg_loss:.4f}  {metric_str}")
        except TypeError:
            auc = dl.evaluate(model, device)
            print(f"[Epoch {ep:03d}] loss={avg_loss:.4f}  AUC={auc:.4f}")


    user_emb = model.embedding_user.weight.detach().cpu().numpy()
    item_emb = model.embedding_item.weight.detach().cpu().numpy()


    user_df = pd.DataFrame(user_emb)
    user_df.insert(0, "user_id", np.arange(len(user_emb)))
    user_df.to_csv("user_embeddings.csv", index=False)

    item_df = pd.DataFrame(item_emb)
    item_df.insert(0, "user_id", np.arange(len(item_emb)))
    item_df.to_csv("item_embeddings.csv", index=False)

    print("✅ Saved user_embeddings.csv and item_embeddings.csv")

if __name__ == "__main__":
    main()
