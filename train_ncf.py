import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from app.data_loader import load_movielens, ImplicitDataset
from app.models.ncf_model import NCF

class EvalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return int(row["userId"]), int(row["movieId"]), float(row["label"])

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for user, pos_item, neg_items in tqdm(loader, desc="Train"):
        user = torch.tensor(user, dtype=torch.long, device=device)
        pos_item = torch.tensor(pos_item, dtype=torch.long, device=device)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)

        optimizer.zero_grad()

        pos_scores = model(user, pos_item)
        pos_labels = torch.ones_like(pos_scores)

        user_rep = user.unsqueeze(1).expand_as(neg_items).reshape(-1)
        neg_items_flat = neg_items.reshape(-1)
        neg_scores = model(user_rep, neg_items_flat)
        neg_labels = torch.zeros_like(neg_scores)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for user, item, label in tqdm(loader, desc="Eval"):
            user = torch.tensor(user, dtype=torch.long, device=device)
            item = torch.tensor(item, dtype=torch.long, device=device)
            scores = model(user, item)
            y_true.extend(label)
            y_pred.extend(scores.cpu().tolist())
    return roc_auc_score(y_true, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_df, test_df, num_users, num_items = load_movielens("data/movielens-1m/ml-1m/ratings.dat")
    train_dataset = ImplicitDataset(train_df, num_items=num_items, num_negatives=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    eval_dataset = EvalDataset(test_df)
    eval_loader = DataLoader(eval_dataset, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(num_users, num_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_auc = 0.0
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        auc = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "models/ncf_model.pth")
            print(f"Saved new best model, AUC={best_auc:.4f}")

if __name__ == "__main__":
    main()
