import gradio as gr
import requests

API_BASE = "http://localhost:5001"

def ui_predict(user_id, item_id):
    r = requests.post(f"{API_BASE}/predict",
                      json={"user_id": int(user_id), "item_id": int(item_id)})
    return r.json().get("score", 0.0)

def ui_recommend(user_id, k):
    r = requests.get(f"{API_BASE}/recommend/{int(user_id)}/{int(k)}")
    data = r.json()
    items = data.get("items", [])
    scores = data.get("scores", [])
    lines = []
    for i, (it, sc) in enumerate(zip(items, scores), 1):
        lines.append(f"{i}. item_id={it}, score={sc:.4f}")
    return "\n".join(lines)

def ui_health():
    r = requests.get(f"{API_BASE}/health")
    return r.json()

with gr.Blocks() as demo:
    gr.Markdown("## Neural Collaborative Filtering Recommender")

    with gr.Tab("Predict"):
        u = gr.Number(label="User ID", value=1, precision=0)
        it = gr.Number(label="Item ID", value=1, precision=0)
        btn_p = gr.Button("Predict")
        out_p = gr.Number(label="Score")
        btn_p.click(ui_predict, [u, it], out_p)

    with gr.Tab("Recommend"):
        u2 = gr.Number(label="User ID", value=1, precision=0)
        k = gr.Number(label="Top-K", value=10, precision=0)
        btn_r = gr.Button("Recommend")
        out_r = gr.Textbox(label="Recommendations")
        btn_r.click(ui_recommend, [u2, k], out_r)

    with gr.Tab("Health"):
        btn_h = gr.Button("Check API")
        out_h = gr.JSON()
        btn_h.click(ui_health, outputs=out_h)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
