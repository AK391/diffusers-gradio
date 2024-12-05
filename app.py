import gradio as gr
import diffusers_gradio

gr.load(
    name='Yuanshi/OminiControl',
    src=diffusers_gradio.registry,
).launch()