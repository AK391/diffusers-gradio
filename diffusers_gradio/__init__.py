import os
import base64
import gradio as gr
import torch
import yaml
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any, Callable, Type
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
from io import BytesIO
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    Transformer2DModelOutput,
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    logger,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)

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
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        mask=None,
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img)
        else:
            self.condition = condition
        assert mask is None, "Mask not supported yet"

    def get_condition(
        self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """Returns the condition image."""
        if condition_type == "depth":
            from transformers import pipeline
            depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cuda",
            )
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
            condition_image = raw_img.convert("RGB").filter(ImageFilter.GaussianBlur(10)).convert("RGB")
            return condition_image
        elif condition_type == "fill":
            return raw_img.convert("RGB")
        return self.condition

    @property
    def type_id(self) -> int:
        """Returns the type id of the condition."""
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """Returns the type id of the condition."""
        return condition_dict[condition_type]

    def _encode_image(self, pipe: FluxPipeline, cond_img: Image.Image) -> torch.Tensor:
        """Encodes an image condition into tokens using the pipeline."""
        cond_img = pipe.image_processor.preprocess(cond_img)
        cond_img = cond_img.to(pipe.device).to(pipe.dtype)
        cond_img = pipe.vae.encode(cond_img).latent_dist.sample()
        cond_img = (cond_img - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        cond_tokens = pipe._pack_latents(cond_img, *cond_img.shape)
        cond_ids = pipe._prepare_latent_image_ids(
            cond_img.shape[0],
            cond_img.shape[2],
            cond_img.shape[3],
            pipe.device,
            pipe.dtype,
        )
        return cond_tokens, cond_ids

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Encodes the condition into tokens, ids and type_id."""
        if self.condition_type in ["depth", "canny", "subject", "coloring", "deblurring", "fill"]:
            tokens, ids = self._encode_image(pipe, self.condition)
        else:
            raise NotImplementedError(f"Condition type {self.condition_type} not implemented")
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        return tokens, ids, type_id

def prepare_params(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    **kwargs: dict,
):
    return (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    )

def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def generate(
    pipeline: FluxPipeline,
    conditions: List[Condition] = None,
    model_config: Optional[Dict[str, Any]] = {},
    condition_scale: float = 1.0,
    **params: dict,
):
    # ... paste the entire generate function here ...

def tranformer_forward(
    transformer: FluxTransformer2DModel,
    condition_latents: torch.Tensor,
    condition_ids: torch.Tensor,
    condition_type_ids: torch.Tensor,
    model_config: Optional[Dict[str, Any]] = {},
    return_conditional_latents: bool = False,
    c_t=0,
    **params: dict,
):
    # ... paste the entire transformer_forward function here ...

class enable_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], activated: bool) -> None:
        self.activated: bool = activated
        if activated:
            return
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]

    def __enter__(self) -> None:
        if self.activated:
            return

        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(0)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self.activated:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]

class set_lora_scale:
    def __init__(self, lora_modules: List[BaseTunerLayer], scale: float) -> None:
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]
        self.scale = scale

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(self.scale)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]

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
    # ... paste the entire attn_forward function here ...

def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):
    # ... paste the entire block_forward function here ...

def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    condition_latents: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    cond_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):
    # ... paste the entire single_block_forward function here ...

def get_fn(model_path: str, **model_kwargs):
    """Create a generation function with the specified model."""
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if model_kwargs.get("lora_weights"):
        pipe.load_lora_weights(
            model_kwargs["lora_weights"],
            weight_name=model_kwargs.get("weight_name"),
            adapter_name=model_kwargs.get("adapter_name")
        )
        
        # Configure attention processing
        pipe.transformer.set_attn_processor(
            lambda attn, *args, **kwargs: attn_forward(
                attn,
                *args,
                model_config=model_kwargs.get("model_config", {}),
                **kwargs
            )
        )
        
        # Configure block processing
        for block in pipe.transformer.transformer_blocks:
            block.forward = lambda *args, **kwargs: block_forward(
                block,
                *args,
                model_config=model_kwargs.get("model_config", {}),
                **kwargs
            )
        
        for block in pipe.transformer.single_transformer_blocks:
            block.forward = lambda *args, **kwargs: single_block_forward(
                block,
                *args,
                model_config=model_kwargs.get("model_config", {}),
                **kwargs
            )
    
    def predict(
        message: str,
        history,
        system_prompt: str,
        condition_type: str,
        temperature: float,
        max_new_tokens: int,
        top_k: int,
        repetition_penalty: float,
        top_p: float,
        lora_scale: float = 1.0,
    ):
        try:
            if isinstance(message, dict) and message.get("files"):
                image = Image.open(message["files"][0]).convert("RGB")
                text = message.get("text", "")
                
                # Process image (center crop and resize)
                w, h, min_size = image.size[0], image.size[1], min(image.size)
                image = image.crop((
                    (w - min_size) // 2,
                    (h - min_size) // 2,
                    (w + min_size) // 2,
                    (h + min_size) // 2,
                ))
                image = image.resize((512, 512))
                
                # Create condition and generate with LoRA control
                condition = Condition(condition_type, image)
                seed_everything(42)  # For reproducibility
                
                with set_lora_scale([pipe.transformer], lora_scale):
                    result = generate(
                        pipe,
                        prompt=text.strip(),
                        conditions=[condition],
                        num_inference_steps=8,
                        height=512,
                        width=512,
                        guidance_scale=7.5,
                        condition_scale=1.0,
                        model_config=model_kwargs.get("model_config", {}),
                    ).images[0]
                
                # Convert to base64 for display
                buffered = BytesIO()
                result.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                yield f"![Generated Image](data:image/png;base64,{img_str})"
            else:
                yield "Please provide both an image and text prompt."
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            yield f"An error occurred: {str(e)}"
            
    return predict

def get_model_path(name: str = None, model_path: str = None) -> str:
    """Get the local path to the model."""
    if model_path:
        return model_path
    
    if name:
        if "/" in name:
            return name
        else:
            model_mapping = {
                "flux": "black-forest-labs/FLUX.1-schnell",
            }
            if name not in model_mapping:
                raise ValueError(f"Unknown model name: {name}")
            return model_mapping[name]
    
    raise ValueError("Either name or model_path must be provided")

def registry(name: str = None, model_path: str = None, **kwargs):
    """Create a Gradio Interface for image generation."""
    model_path = get_model_path(name, model_path)
    fn = get_fn(model_path, **kwargs)

    return gr.ChatInterface(
        fn=fn,
        multimodal=True,
        additional_inputs_accordion=gr.Accordion("⚙️ Parameters", open=False),
        additional_inputs=[
            gr.Textbox(
                "You are a helpful AI assistant.",
                label="System prompt"
            ),
            gr.Dropdown(
                choices=list(condition_dict.keys()),
                value="subject",
                label="Condition Type"
            ),
            gr.Slider(0, 1, 0.7, label="Temperature"),
            gr.Slider(128, 4096, 1024, label="Max new tokens"),
            gr.Slider(1, 80, 40, label="Top K sampling"),
            gr.Slider(0, 2, 1.1, label="Repetition penalty"),
            gr.Slider(0, 1, 0.95, label="Top P sampling"),
            gr.Slider(0, 2, 1.0, label="LoRA scale"),
        ],
    )