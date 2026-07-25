import gradio as gr
from pathlib import Path

def load_poster():
    poster_path = Path("poster.html")
    if poster_path.exists():
        return poster_path.read_text()
    return "<h1>Poster loading error</h1>"

with gr.Blocks(title="Mechanistic Data Attribution - ICML 2026 Repro") as demo:
    gr.HTML(load_poster())

if __name__ == "__main__":
    demo.launch()
