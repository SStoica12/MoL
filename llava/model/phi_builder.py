"""Phi loading and compatibility with the released MoL/Bunny LoRA adapters."""
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, PhiConfig
from transformers.utils.hub import cached_file

from .language_model.llava_phi import RouterLlavaPhiForCausalLM


def is_phi_config(config):
    return config is not None and getattr(config, "model_type", None) == "phi"


def checkpoint_file(model_path, filename, required=True):
    if Path(model_path).is_dir():
        path = Path(model_path) / filename
        if path.is_file():
            return str(path)
        if required:
            raise FileNotFoundError(path)
        return None
    return cached_file(
        model_path, filename, _raise_exceptions_for_missing_entries=required
    )


def normalize_phi_adapter_state(state):
    """Map legacy `router.*` and current PEFT exports to registered model keys."""
    normalized = {}
    for key, value in state.items():
        if key.startswith("base_model.model."):
            key = key[len("base_model.model."):]
        elif key.startswith("model.model."):
            key = key[len("model."):]
        if key.startswith("router."):
            key = "router_clip." + key[len("router."):]
        if key == "router_clip.weights":
            key = "router_clip._weight_fn.weights"
        if key in ("router_clip.hybrid_reserve_d_weight", "router_clip.hybrid_reserve_d_bias"):
            if value.numel() != 1:
                raise ValueError(f"Expected one scalar for {key}")
            value = value.reshape(())
        if key in normalized:
            raise ValueError(f"Duplicate Phi adapter key after normalization: {key}")
        normalized[key] = value
    return normalized


def restore_phi_adapter(model, state):
    state = normalize_phi_adapter_state(state)
    expected = model.state_dict()
    required = {k for k in expected if k.startswith(("model.mm_projector.", "router_clip."))}
    missing = required - state.keys()
    unexpected = state.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(
            f"Incomplete or incompatible Phi multimodal adapter: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    # Backbone tensors come from the base LM; every projector/router tensor must
    # be present above. PyTorch also verifies the shape of every supplied tensor.
    model.load_state_dict(state, strict=False)


def configure_phi_tokens(model, tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name)
        if value is not None:
            setattr(model.config, name, value)
            setattr(model.generation_config, name, value)


def load_phi_model(model_path, model_base, config, *, device, model_kwargs,
                   num_layers_to_choose=None):
    """Return the same five values as the shared MoL inference loader."""
    config = PhiConfig.from_dict(config.to_dict())
    config.encoder_type = getattr(config, "encoder_type", "siglip")
    if config.encoder_type != "siglip":
        raise ValueError("Phi checkpoints currently support a single SigLIP encoder.")
    if not getattr(config, "mm_vision_tower", None):
        raise ValueError("Phi checkpoint is missing its multimodal vision tower configuration.")
    if num_layers_to_choose is not None:
        if num_layers_to_choose < 1:
            raise ValueError("num_layers_to_choose must be positive.")
        config.num_layers_to_choose = num_layers_to_choose

    adapter_path = checkpoint_file(model_path, "adapter_config.json", required=False)
    if adapter_path:
        adapter_config = json.loads(Path(adapter_path).read_text())
        model_base = model_base or adapter_config.get("base_model_name_or_path")
        if not model_base:
            raise ValueError("Phi LoRA loading requires its base language model.")
    tokenizer_source = model_base if adapter_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    kwargs = dict(model_kwargs)
    if device == "cpu":
        kwargs["torch_dtype"] = torch.float32
    model, loading = RouterLlavaPhiForCausalLM.from_pretrained(
        model_base if adapter_path else model_path,
        config=config, low_cpu_mem_usage=True, output_loading_info=True,
        initialize_router=not bool(adapter_path), **kwargs,
    )
    if adapter_path:
        # Transformers 4.37 cannot materialize missing scalar parameters from
        # meta tensors. Build the hybrid router after the base LM is loaded.
        model.initialize_router()
        weights_path = checkpoint_file(model_path, "non_lora_trainables.bin")
        restore_phi_adapter(model, torch.load(weights_path, map_location="cpu"))
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
    else:
        missing = [k for k in loading["missing_keys"]
                   if k.startswith(("model.mm_projector.", "router_clip."))]
        if missing:
            raise ValueError(f"Phi checkpoint is missing multimodal weights: {missing}")

    # Phi pads its vocabulary to 51200. Preserve the pretrained embedding/head
    # sizes rather than shrinking them to the tokenizer's vocabulary length.
    configure_phi_tokens(model, tokenizer)
    tower = model.get_vision_tower()
    if not tower.is_loaded:
        tower.load_model()
    embedding = model.get_input_embeddings().weight
    tower.to(device=embedding.device, dtype=embedding.dtype)
    model.router_clip.to(device=embedding.device, dtype=embedding.dtype)
    model.eval()
    return tokenizer, model, None, tower.image_processor, getattr(
        config, "max_sequence_length", config.max_position_embeddings
    )
