from flask import Flask, request, jsonify
import torch
import os

from app.models.ncf_model import NCF
from app.data_loader import load_movielens

app = Flask(__name__)

_, _, NUM_USERS, NUM_ITEMS = load_movielens("data/movielens-1m/ml-1m/ratings.dat")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NCF(NUM_USERS, NUM_ITEMS)
state_path = "models/ncf_model.pth"
if os.path.exists(state_path):
    model.load_state_dict(torch.load(state_path, map_location=device))
model.to(device)
model.eval()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device)})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_id = int(data["user_id"])
    item_id = int(data["item_id"])

    with torch.no_grad():
        user = torch.tensor([user_id], dtype=torch.long, device=device)
        item = torch.tensor([item_id], dtype=torch.long, device=device)
        score = model(user, item).item()
    return jsonify({"user_id": user_id, "item_id": item_id, "score": score})

@app.route("/recommend/<int:user_id>/<int:k>", methods=["GET"])
def recommend(user_id, k):
    with torch.no_grad():
        items = torch.arange(1, NUM_ITEMS + 1, dtype=torch.long, device=device)
        user = torch.full_like(items, fill_value=user_id, dtype=torch.long, device=device)
        scores = model(user, items)
        topk = torch.topk(scores, k)
        indices = topk.indices.cpu().tolist()
        values = topk.values.cpu().tolist()
    return jsonify({"user_id": user_id, "items": indices, "scores": values})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
