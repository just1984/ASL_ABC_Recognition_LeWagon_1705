import gradio as gr
from model import predict

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.JSON()]
)

iface.launch()
