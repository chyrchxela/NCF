import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class ImplicitDataset(Dataset):
    def __init__(self, df, num_items, num_negatives=4):
        self.df = df.reset_index(drop=True)
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_pos_items = (
            self.df[self.df["label"] == 1]
            .groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user = int(row["userId"])
        pos_item = int(row["movieId"])

        negatives = []
        pos_set = self.user_pos_items.get(user, set())
        while len(negatives) < self.num_negatives:
            neg_item = np.random.randint(1, self.num_items + 1)
            if neg_item not in pos_set:
                negatives.append(neg_item)

        return user, pos_item, np.array(negatives, dtype=np.int64)

def load_movielens(path="data/movielens-1m/ml-1m/ratings.dat"):
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    ratings["label"] = (ratings["rating"] >= 4).astype(int)
    ratings = ratings[["userId", "movieId", "label"]]

    user_ids = {u: i + 1 for i, u in enumerate(ratings["userId"].unique())}
    item_ids = {m: i + 1 for i, m in enumerate(ratings["movieId"].unique())}
    ratings["userId"] = ratings["userId"].map(user_ids)
    ratings["movieId"] = ratings["movieId"].map(item_ids)

    num_users = len(user_ids)
    num_items = len(item_ids)

    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), num_users, num_items
