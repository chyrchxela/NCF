import gradio as gr

print("Starting Gradio...")

def add(a, b):
    return a + b

demo = gr.Interface(fn=add, inputs=["number", "number"], outputs="number")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
