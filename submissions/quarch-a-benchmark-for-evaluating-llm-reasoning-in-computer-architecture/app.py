import gradio as gr
import os

def load_page(filename):
    path = os.path.join("pages", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Page not found."

with gr.Blocks(title="QuArch Reproduction Space") as demo:
    gr.Markdown("# QuArch Benchmark Reproduction")
    overview = load_page("01_overview.md")
    gr.Markdown(overview)

if __name__ == "__main__":
    demo.launch()
