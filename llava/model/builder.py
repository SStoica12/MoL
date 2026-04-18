#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import re
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import json
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from safetensors.torch import load_file
from llava.model.language_model.llava_llama import RouterLlavaLlamaForCausalLM, DoubleRouterLlavaLlamaForCausalLM
from llava.model.language_model.llava_mpt import RouterLlavaMptForCausalLM, DoubleRouterLlavaMptForCausalLM
from llava.model.router import Router, TextConditionedRouter

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

def load_pretrained_model_router(model_path, model_base, model_name, 
                                mm_use_text_enc,
                                text_pool,
                                text_scale,
                                text_conditioning,
                                text_attention_type, # "concat" or "qk"
                                text_attention_dim,
                                attn_level,
                                force_reserve_layer,
                                learnable_gate,
                                confidence_gate,
                                confidence_gate_threshold,
                                confidence_gate_slope,
                                load_8bit=False, load_4bit=False, 
                                device_map="auto", device="cuda", 
                                use_flash_attn=False, num_layers_to_choose=1, 
                                router_alpha=0.01, encoder_type ="dinov2_txt", 
                                fusion_type= "cross_attn", 
                                **kwargs):
    rank0_print(f"\n      num_layers_to_choose is {num_layers_to_choose} and router_alpha is {0.01} and device_map {device_map}")
    rank0_print("\n model_base is None : ", model_base is None)
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    reserve_layer = kwargs.pop('reserve_layer', None) # this is only used for router, not needed for model loading and would cause error if we pass it in
    rank0_print("\n model_name : ", model_name)
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            lora_cfg_pretrained.mm_hidden_size_dino = 1024
            lora_cfg_pretrained.fusion_type = fusion_type # this only matters if we have clip + dino instance

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print('Loading LLaVA from base model...')
            if "clip" in encoder_type and "dino" in encoder_type:
                model = DoubleRouterLlavaLlamaForCausalLM().from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            else:
                model = RouterLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                model.encoder_type = encoder_type
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            rank0_print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            rank0_print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print('Merging LoRA weights...')
            model = model.merge_and_unload()
            rank0_print('Model is loaded...')
        
        elif model_base is not None:
            # this may be mm projector only
            rank0_print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                if "clip" in encoder_type and "dino" in encoder_type:
                    model = DoubleRouterLlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = RouterLlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    model.encoder_type = encoder_type
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                cfg_pretrained.mm_hidden_size_dino = 1024
                cfg_pretrained.fusion_type = fusion_type
                # cfg_pretrained.mm_hidden_size_dino = 1024
                # cfg_pretrained.fusion_type = fusion_type # this only matters if we have clip + dino instance
                if "clip" in encoder_type and "dino" in encoder_type:
                    model = DoubleRouterLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = RouterLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    model.encoder_type = encoder_type
            if "clip" in encoder_type:
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            if "dino" in encoder_type:
                dino_projector_weights = torch.load(os.path.join(model_path, 'dino_projector.bin'), map_location='cpu')
                dino_projector_weights = {k: v.to(torch.float16) for k, v in dino_projector_weights.items()}
                model.load_state_dict(dino_projector_weights, strict=False)
        else:
            import transformers
            rank0_print("\n kwargs: ", kwargs)
            # can not use our config or get: AttributeError: 'LlavaConfig' object has no attribute 'attention_dropout'
            cfg_pretrained = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            cfg_pretrained.mm_hidden_size_dino = 1024
            cfg_pretrained.fusion_type = fusion_type
            cfg_pretrained.attention_bias = False
            cfg_pretrained.attention_dropout = 0.0
            cfg_pretrained.rope_theta = 10000.0
            cfg_pretrained.encoder_type = encoder_type
            cfg_pretrained.mm_use_text_enc = mm_use_text_enc
            cfg_pretrained.mm_vision_select_layer = -2
            rank0_print("config: ", cfg_pretrained)
            # import sys; sys.exit(0)
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                if "clip" in encoder_type and "dino" in encoder_type:
                    model = DoubleRouterLlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = RouterLlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                    model.encoder_type = encoder_type
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if "clip" in encoder_type and "dino" in encoder_type:
                    model = DoubleRouterLlavaMptForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        **kwargs
                    )
                else:
                    model = RouterLlavaMptForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        **kwargs
                    )
                    model.encoder_type = encoder_type
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if "clip" in encoder_type and "dino" in encoder_type:
                    model = DoubleRouterLlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        **kwargs
                    )
                else:
                    rank0_print("\n kwargs :", kwargs)
                    model = RouterLlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        config=cfg_pretrained,
                        **kwargs
                    )
                    model.encoder_type = encoder_type
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            rank0_print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print(f"Merging weights")
            model = model.merge_and_unload()
            rank0_print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor_dino = None
    image_processor_clip = None
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        # rank0_print("\n type model: ", type(model))
        # rank0_print("\n model : ", model)
        # rank0_print("\n model.config : ", model.config)
        # import sys; sys.exit(0)
        if "dino" in model.config.encoder_type:
            dino_tower = model.model.dino_tower
            if not dino_tower.is_loaded:
                dino_tower.load_model(device_map=device_map)
            dino_tower.to(device=device, dtype=torch.float16)
            image_processor_dino = dino_tower.image_processor
        if "clip" in model.config.encoder_type:
            vision_tower = model.model.vision_tower
            vision_tower.load_model(device_map=device_map)
            if not vision_tower.is_loaded:
                vision_tower.load_model(device_map=device_map)
            vision_tower.to(device=device, dtype=torch.float16)
            image_processor_clip = vision_tower.image_processor

        if "dino" in model.config.encoder_type: # initialize router_dino_weights
            image_size = dino_tower.config.image_size
            patch_size = dino_tower.config.patch_size
            hidden_size = dino_tower.config.hidden_size_dino
            num_image_tokens = (image_size // patch_size)**2 # 257 even for dino txt, which has 4 registers
            router_dino_weights = torch.zeros((num_image_tokens, hidden_size)).to(dtype=dino_tower.dtype, device=dino_tower.device)
            rank0_print("\n router_dino_weights : ", router_dino_weights.shape)
            # import sys; sys.exit(0)
            assert router_dino_weights.is_cuda
            if not mm_use_text_enc:
                rank0_print("\n We using normal Router")
                model.router_dino = Router(model.config, router_dino_weights, model.config.mm_vision_select_feature, router_alpha)
            else:
                rank0_print("\n We are using TextConditionedRouter")
                model.router_dino = TextConditionedRouter(config=model.config, weights=router_dino_weights,
                                                        load_balancing_alpha=router_alpha,
                                                        reserve_layer = reserve_layer if reserve_layer is not None else dino_tower.config.mm_vision_select_feature,
                                                        top_k=num_layers_to_choose,
                                                        text_pool=text_pool,
                                                        text_scale=text_scale, # overall influence of text. Would this allow us to better modulate the features. hyperparam sweep
                                                        text_conditioning=text_conditioning, # "film" or "attention"
                                                        text_attention_type=text_attention_type, # "concat" or "qk"
                                                        text_attention_dim=text_attention_dim,
                                                        attn_level=attn_level, # "token" or "layer" (used when text_conditioning == "attention")
                                                        force_reserve_layer=force_reserve_layer,
                                                        learnable_gate=learnable_gate,
                                                        confidence_gate=confidence_gate,
                                                        confidence_gate_threshold=confidence_gate_threshold,
                                                        confidence_gate_slope=confidence_gate_slope)
            
            # rank0_print("dino_tower dtype", dino_tower.dtype)
            # import sys; sys.exit(0)
            model.router_dino.to(device=router_dino_weights.device, dtype=dino_tower.dtype) # for some reason if you do not do this reserve_gate, text_attn_k,  text_attn_q weights are on cpu
            # rank0_print('router_dino._weight_fn.weights', model.router_dino._weight_fn.weights.device)
            # rank0_print('router_dino.reserve_gate.weight', model.router_dino.reserve_gate.weight.device,)
            # rank0_print( 'router_dino.text_attn_k.weight', model.router_dino.text_attn_k.weight.device,)
            # rank0_print('router_dino.text_attn_q.weight', model.router_dino.text_attn_q.weight.device)

            # import pdb; pdb.set_trace()
            model.router_dino.requires_grad_(False) # we need to learn the router
        if "clip" in model.config.encoder_type:
            image_size = vision_tower.vision_tower.config.image_size
            patch_size = vision_tower.vision_tower.config.patch_size
            hidden_size = vision_tower.vision_tower.config.hidden_size
            num_image_tokens = (image_size // patch_size)**2 + 1 # 257 even for dino txt, which has 4 registers
            router_clip_weights = torch.zeros((num_image_tokens, hidden_size)).to(dtype=vision_tower.dtype, device=vision_tower.device)
            assert router_clip_weights.is_cuda
            if not mm_use_text_enc:
                rank0_print("\n We using normal Router")
                model.router_clip = Router(model.config, router_dino_weights, model.config.mm_vision_select_feature, router_alpha)
            else:
                rank0_print("\n We using TextConditionedRouter")
                model.router_clip = TextConditionedRouter(config=model.config, weights=router_clip_weights,
                                                        load_balancing_alpha=router_alpha,
                                                        reserve_layer = dino_tower.config.mm_vision_select_feature,
                                                        top_k=num_layers_to_choose,
                                                        text_pool=text_pool,
                                                        text_scale=text_scale, # overall influence of text. Would this allow us to better modulate the features. hyperparam sweep
                                                        text_conditioning=text_conditioning, # "film" or "attention"
                                                        text_attention_type=text_attention_type, # "concat" or "qk"
                                                        text_attention_dim=text_attention_dim,
                                                        attn_level=attn_level, # "token" or "layer" (used when text_conditioning == "attention")
                                                        force_reserve_layer=force_reserve_layer,
                                                        learnable_gate=learnable_gate,
                                                        confidence_gate=confidence_gate,
                                                        confidence_gate_threshold=confidence_gate_threshold,
                                                        confidence_gate_slope=confidence_gate_slope)
            model.router_clip.to(device=router_dino_weights.device, dtype=vision_tower.dtype)
            model.router_clip.requires_grad_(False) # we need to learn the router
       
        # overwrite the router weights
        rank0_print("♻️  Overwriting Router weights with custom finetuned weights...")
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing: {index_path}")

        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        # find router key in the big model checkpoint
        router_keys_dino = [k for k in weight_map.keys() if "router_dino" in k.lower() and ("weight" in k.lower())]
        router_keys_clip = [k for k in weight_map.keys() if "router_clip" in k.lower() and ("weight" in k.lower())]
        if router_keys_dino:
            rank0_print("router_keys_dino : ", router_keys_dino)
            shard_file = weight_map[router_keys_dino[0]]
            shard_path = os.path.join(model_path, shard_file)
            rank0_print(f"\n    Loaded checkpoint path {shard_path}")
            shard_weights = safe_load_file(shard_path)  # loads CPU tensors
            router_state_dict_dino = {}
            for key, value in shard_weights.items():
                    # We filter for keys belonging to the router
                    if "router_dino" in key:
                        key = key.replace("router_dino.", "")  # need just weights
                        # assert key == "weights"
                        router_state_dict_dino[key] = value
            #    strict=False is crucial because we are only loading the vision subset
            if len(router_state_dict_dino) > 0:
                missing, unexpected = model.router_dino.load_state_dict(router_state_dict_dino, strict=False)
                if len(missing) != 0 or len(unexpected) != 0:
                    rank0_print("missing: ", missing)
                    rank0_print("unexpected: ", unexpected)
                    # rank0_print(missing, unexpected)
                    import sys; sys.exit(0)
                else:
                    rank0_print(f"✅ Successfully restored {len(router_state_dict_dino)} custom router keys.")
            else:
                rank0_print("⚠️  Warning: No custom router keys found in shards.")

        if router_keys_clip:
            rank0_print("router_keys_clip : ", router_keys_clip)
            shard_file = weight_map[router_keys_clip[0]]
            shard_path = os.path.join(model_path, shard_file)
            rank0_print(f"\n    Loaded checkpoint path {shard_path}")
            shard_weights = safe_load_file(shard_path)  # loads CPU tensors
            router_state_dict_clip = {}
            for key, value in shard_weights.items():
                    # We filter for keys belonging to the router
                    if "router_clip" in key:
                        key = key.replace("router_clip.", "")  # need just weights
                        # assert key == "weights"
                        router_state_dict_clip[key] = value
            #    strict=False is crucial because we are only loading the vision subset
            if len(router_state_dict_clip) > 0:
                missing, unexpected = model.router_clip.load_state_dict(router_state_dict_clip, strict=False)
                if len(missing) != 0 or len(unexpected) != 0:
                    rank0_print("missing: ", missing)
                    rank0_print("unexpected: ", unexpected)
                    # rank0_print(missing, unexpected)
                    import sys; sys.exit(0)
                else:
                    rank0_print(f"✅ Successfully restored {len(router_state_dict_clip)} custom router keys.")
            else:
                rank0_print("⚠️  Warning: No custom router keys found in shards.")
       
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor_dino, image_processor_clip, context_len


def load_pretrained_model(model_path, model_base, model_name, encoder_type ="dinov2_txt", load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    # import pdb; pdb.set_trace()
    if 'llava' in model_name.lower():
        # Load LLaVA model
        # import pdb; pdb.set_trace()
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            rank0_print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            rank0_print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print('Merging LoRA weights...')
            model = model.merge_and_unload()
            rank0_print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            rank0_print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            import transformers
            cfg_pretrained = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            cfg_pretrained.mm_hidden_size_dino = 1024
            cfg_pretrained.attention_bias = False
            cfg_pretrained.attention_dropout = 0.0
            cfg_pretrained.rope_theta = 10000.0
            cfg_pretrained.encoder_type = encoder_type
            cfg_pretrained.mm_vision_select_layer = -2

            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    config=cfg_pretrained,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    config=cfg_pretrained,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            rank0_print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print(f"Merging weights")
            model = model.merge_and_unload()
            rank0_print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor_clip = None
    image_processor_dino = None
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()

        

        # import pdb; pdb.set_trace()
        if "dino" in model.config.encoder_type:
            dino_tower = model.model.dino_tower
            if not dino_tower.is_loaded:
                dino_tower.load_model(device_map=device_map)
            dino_tower.to(device=device, dtype=torch.float16)
            image_processor_dino = dino_tower.image_processor
        if "clip" in model.config.encoder_type:
            vision_tower = model.model.vision_tower
            vision_tower.load_model(device_map=device_map)
            if not vision_tower.is_loaded:
                vision_tower.load_model(device_map=device_map)
            vision_tower.to(device=device, dtype=torch.float16)
            image_processor_clip = vision_tower.image_processor

            # rank0_print("♻️  Overwriting vanilla vision tower with custom finetuned weights...")
            # # import pdb; pdb.set_trace()
            # import glob
            # from safetensors.torch import load_file
            
            # # Locate all safetensor shards
            # shard_files = glob.glob(os.path.join(model_path, "*.safetensors"))
            
            # vision_state_dict = {}
            
            # # 2. Iterate through shards and collect ONLY vision tower keys
            # for shard in shard_files:
            #     shard_weights = load_file(shard)
            #     for key, value in shard_weights.items():
            #         # We filter for keys belonging to the vision tower
            #         if "vision_tower" in key:
            #             vision_state_dict[key] = value
            
            # # 3. Load these weights into the model
            # #    strict=False is crucial because we are only loading the vision subset
            # if len(vision_state_dict) > 0:
            #     model.load_state_dict(vision_state_dict, strict=False)
            #     rank0_print(f"✅ Successfully restored {len(vision_state_dict)} custom vision tower keys.")
            # else:
            #     rank0_print("⚠️  Warning: No custom vision tower weights found in shards.")

        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        
        # image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    # import pdb; pdb.set_trace()
    return tokenizer, model, image_processor_dino, image_processor_clip, context_len