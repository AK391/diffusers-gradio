import os
import base64
import gradio as gr
import torch
import yaml
import numpy as np
import cv2
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from PIL import Image, ImageFilter
from io import BytesIO
from diffusers.pipelines import FluxPipeline
from diffusers.models.attention_processor import Attention, F
from peft.tuners.tuners_utils import BaseTunerLayer
import subprocess
from huggingface_hub import hf_hub_download, list_repo_files
from diffusers import DiffusionPipeline
from threading import Thread

__version__ = "0.0.1"

condition_dict = {
    "depth": 0,
    "canny": 1,
    "subject": 4,
    "coloring": 6,
    "deblurring": 7,
    "fill": 9,
}

class Condition(object):
    def __init__(self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor] = None):
        self.condition_type = condition_type
        assert raw_img is not None
        self.condition = self.get_condition(condition_type, raw_img)

    def get_condition(self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """Returns the condition image."""
        if condition_type == "depth":
            from transformers import pipeline
            depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device="cuda")
            source_image = raw_img.convert("RGB")
            condition_img = depth_pipe(source_image)["depth"].convert("RGB")
            return condition_img
        elif condition_type == "canny":
            img = np.array(raw_img)
            edges = cv2.Canny(img, 100, 200)
            edges = Image.fromarray(edges).convert("RGB")
            return edges
        elif condition_type == "subject":
            return raw_img
        elif condition_type == "coloring":
            return raw_img.convert("L").convert("RGB")
        elif condition_type == "deblurring":
            return raw_img.convert("RGB").filter(ImageFilter.GaussianBlur(10)).convert("RGB")
        elif condition_type == "fill":
            return raw_img.convert("RGB")
        return raw_img

    @property
    def type_id(self) -> int:
        """Returns the type id of the condition."""
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """Returns the type id of the condition."""
        return condition_dict[condition_type]

# Attention forward function
def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    with enable_lora((attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if encoder_hidden_states is not None:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if condition_latents is not None:
        query = torch.cat([query, cond_query], dim=2)
        key = torch.cat([key, cond_key], dim=2)
        value = torch.cat([value, cond_value], dim=2)

    # Attention mask handling
    if not model_config.get("union_cond_attn", True):
        attention_mask = torch.ones(query.shape[2], key.shape[2], device=query.device, dtype=torch.bool)
        condition_n = cond_query.shape[2]
        attention_mask[-condition_n:, :-condition_n] = False
        attention_mask[:-condition_n, -condition_n:] = False

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    return hidden_states

# Block forward function
def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    model_config: Optional[Dict[str, Any]] = {},
):
    use_cond = condition_latents is not None
    with enable_lora((self.norm1.linear,), model_config.get("latent_lora", False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

    if use_cond:
        norm_condition_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = self.norm1(condition_latents, emb=cond_temb)

    # Attention
    result = attn_forward(
        self.attn,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    if use_cond:
        cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
        condition_latents = condition_latents + cond_attn_output

    # LayerNorm + MLP
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    if use_cond:
        norm_condition_latents = self.norm2(condition_latents)
        norm_condition_latents = norm_condition_latents * (1 + cond_scale_mlp[:, None]) + cond_shift_mlp[:, None]

    # Feed-forward
    with enable_lora((self.ff.net[2],), model_config.get("latent_lora", False)):
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output

    if use_cond:
        condition_latents = condition_latents + cond_ff_output

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None

# Single block forward function
def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    condition_latents: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    model_config: Optional[Dict[str, Any]] = {},
):
    using_cond = condition_latents is not None
    residual = hidden_states
    with enable_lora((self.norm.linear, self.proj_mlp), model_config.get("latent_lora", False)):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = self.norm(condition_latents, emb=cond_temb)
        mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_condition_latents))

    attn_output = attn_forward(
        self.attn,
        hidden_states=norm_hidden_states,
        condition_latents=norm_condition_latents if using_cond else None,
    )

    if using_cond:
        attn_output, cond_attn_output = attn_output

    with enable_lora((self.proj_out,), model_config.get("latent_lora", False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states) + residual

    if using_cond:
        condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
        cond_gate = cond_gate.unsqueeze(1)
        condition_latents = cond_gate * self.proj_out(condition_latents) + residual_cond

    return hidden_states if not using_cond else (hidden_states, condition_latents)

# Initialize the pipeline
def init_pipeline():
    global pipe
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="omini/subject_512.safetensors", adapter_name="subject")

# Process image and text
def process_image_and_text(image, text):
    # Center crop image
    w, h, min_size = image.size[0], image.size[1], min(image.size)
    image = image.crop(((w - min_size) // 2, (h - min_size) // 2, (w + min_size) // 2, (h + min_size) // 2))
    image = image.resize((512, 512))

    condition = Condition("subject", image)

    if pipe is None:
        init_pipeline()

    result_img = generate(
        pipe,
        prompt=text.strip(),
        conditions=[condition],
        num_inference_steps=8,
        height=512,
        width=512,
    ).images[0]

    return result_img

# Gradio interface

demo = gr.ChatInterface(
    fn=process_image_and_text,  # Function to handle chat responses
    type="messages",             # Message type
    multimodal=True,             # Enable multimodal input
    title="OminiControl / Subject driven generation",
    examples=["hello", "hola", "merhaba"],  # Example inputs
)


def register_model(name: str, model_loader: Callable):
    """Register a model in the registry."""
    registry[name] = model_loader

def get_model_path(name: str, model_path: str = None) -> str:
    """Retrieve the model path based on the name or provided path."""
    # Implement logic to get the model path
    return model_path or f"models/{name}"

def get_fn(model_path: str, **model_kwargs):
    """Create a chat function with the specified model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the diffusion model
    model = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

    def predict(
        message: str,
        history,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_k: int,
        repetition_penalty: float,
        top_p: float
    ):
        try:
            # Handle input format
            if isinstance(message, dict):
                text = message.get("text", "")
                files = message.get("files", [])
                
                # Process images
                images = [Image.open(file).convert("RGB") for file in files] if files else []
            else:
                text = message
                images = []

            # Input validation
            if text == "" and not images:
                raise gr.Error("Please input a query and optionally image(s).")

            # Generate output using the diffusion model
            result_img = model(prompt=text, images=images, num_inference_steps=8).images[0]

            return result_img

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            yield f"An error occurred: {str(e)}"

    return predict

def registry(name: str = None, model_path: str = None, **kwargs):
    """Create a Gradio Interface with similar styling and parameters."""
    
    model_path = get_model_path(name, model_path)
    fn = get_fn(model_path, **kwargs)

    interface = gr.ChatInterface(
        fn=fn,
        multimodal=True,  # Enable multimodal input
        additional_inputs_accordion=gr.Accordion("⚙️ Parameters", open=False),
        additional_inputs=[
            gr.Textbox(
                "You are a helpful AI assistant.",
                label="System prompt"
            ),
            gr.Slider(0, 1, 0.7, label="Temperature"),
            gr.Slider(128, 4096, 1024, label="Max new tokens"),
            gr.Slider(1, 80, 40, label="Top K sampling"),
            gr.Slider(0, 2, 1.1, label="Repetition penalty"),
            gr.Slider(0, 1, 0.95, label="Top P sampling"),
        ],
    )
    
    return interface

# Example usage of the registry function
if __name__ == "__main__":
    interface = registry(name='Yuanshi/OminiControl')  # Specify your model name or path
    interface.launch(debug=True, ssr_mode=False)