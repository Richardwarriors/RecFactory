from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn
from PData.dataset import DatasetSplitter


class RatingDataset(Dataset):
    def __init__(self, df):
        self.user = torch.LongTensor(df['userid'].values)
        self.item = torch.LongTensor(df['itemid'].values)
        self.rating = torch.FloatTensor(df['rating'].values)
    def __len__(self): return len(self.user)
    def __getitem__(self, idx): return self.user[idx], self.item[idx], self.rating[idx]

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim = 100):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, user, item):
        p_u,p_i = self.user_embedding(user), self.item_embedding(item)
        b_u,b_i = self.user_bias(user), self.item_bias(item)
        return p_u.mul(p_i).sum(dim=1) + b_u.squeeze() + b_i.squeeze() + self.global_bias


def train(model, train_loader, val_loader, test_loader, lr=0.01, epochs=120):
    #check GPU state
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU) as device")
    else:
        device = torch.device("cpu")
        print("Using CPU as device")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u,i)
            loos = loss_fn(pred,r)
            optimizer.zero_grad()
            loos.backward()
            optimizer.step()
            train_loss += loos.item() * len(r)
        train_rmse = (train_loss / len(train_loader.dataset)) ** 0.5

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i)
                val_loss += ((pred - r) ** 2).sum().item()
        val_rmse = (val_loss / len(val_loader.dataset)) ** 0.5

        print(f"Epoch {epoch}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    PATH = "mf_best.pth"
    torch.save(best_state, PATH)
    print(f"Best model saved to {PATH}")

    print("Best validation RMSE:", best_val_rmse)
    model.load_state_dict(best_state)
    model = model.to(device).eval()

    test_loss = 0.0
    with torch.no_grad():
        for u, i, r in test_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            test_loss += ((pred - r) ** 2).sum().item()
    test_rmse = (test_loss / len(test_loader.dataset)) ** 0.5
    print("Final Test RMSE:", test_rmse)
    return model, test_rmse

if __name__ == "__main__":
    splitter = DatasetSplitter(filepath='../../OData/ml_1M/ratings.dat', sep='::')
    splitter.load_data()
    splitter.split_member_nonmember()
    splitter.split_train_val_test()
    

    train_df = splitter.train_data
    val_df = splitter.val_data
    test_df = splitter.test_data
    
    train_loader = DataLoader(RatingDataset(train_df), batch_size=512, shuffle=True)
    val_loader   = DataLoader(RatingDataset(val_df), batch_size=512, shuffle=False)
    test_loader  = DataLoader(RatingDataset(test_df), batch_size=512, shuffle=False)

    num_users = train_df['userid'].nunique()
    num_items = train_df['itemid'].nunique()
    print(f"Number of users: {num_users}, Number of items: {num_items}")

    model = MatrixFactorization(num_users, num_items, embedding_dim=100)
    trained_model, final_rmse = train(model, train_loader, val_loader, test_loader)
