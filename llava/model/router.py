import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import json

class Router(nn.Module):
    def __init__(self, config, weights, top_k = 1, 
                load_balancing_alpha=0.01, force_reserve_layer=False, 
                reserve_layer=-2, save_router_stats=True, save_steps_router_stats=200, 
                cap_softmax=False, cap_softmax_value=0.06,
                learnable_gate=False, confidence_gate=False,
                confidence_gate_threshold=0.6, confidence_gate_slope=10.0,
                vision_feature_select_strategy="cls_patch"):
        super().__init__()
        self.num_layers_to_choose = top_k
        self.config = config
        self.num_layers = None
        self.vision_feature_select_strategy = vision_feature_select_strategy  # "patch", "cls_patch", "cls"
        self.weights = nn.Parameter(weights) #[num_tokens, hidden_size]
        self.load_balancing_alpha = load_balancing_alpha
        self.router_temperature = 1.0
        self.force_reserve_layer = force_reserve_layer
        self.reserve_layer = reserve_layer
        self.save_router_stats = save_router_stats
        self.save_steps_router_stats = save_steps_router_stats
        self.cap_softmax = cap_softmax
        self.cap_softmax_value = cap_softmax_value
        self.learnable_gate = learnable_gate
        self.confidence_gate = confidence_gate
        self.confidence_gate_threshold = confidence_gate_threshold
        self.confidence_gate_slope = confidence_gate_slope
        self.reserve_gate = nn.Linear(weights.shape[1], 1) if learnable_gate else None
        # Cache router probabilities from the most recent forward pass so auxiliary
        # losses can reuse the same graph without triggering a second router pass.
        self._latest_router_probs_for_aux = None
        # Cache the latest routing decision during inference so eval writers can
        # attach per-question layer selections to their output rows.
        self._latest_routing_decision = None
        self.hybrid_debug_per_step = {}
        if self.learnable_gate and self.confidence_gate:
            raise ValueError("learnable_gate and confidence_gate are mutually exclusive.")
        # stats 
        if save_router_stats and self.training:
            self.router_probs_per_step = {}
            self.step_rp = 0
            self.expert_mask_per_step = {}
            self.step_ep = 0

        # self.bias = nn.Parameter(torch.zeros(24)) # typically not used
        # usually MoE have a capacity parameter which restricts how many tokens a expert gets, but 
        # do not do that here, so I removed it

    def _select_vision_features(self, image_features: torch.Tensor) -> torch.Tensor:
        strategy = self.vision_feature_select_strategy
        if strategy in ("default", "cls_patch"):
            return image_features
        if strategy == "patch":
            return image_features[:, 1:, :]
        if strategy == "cls":
            return image_features[:, :1, :]
        raise ValueError(f"Unexpected select feature: {strategy}")

    def _resolve_reserve_idx(self, num_layers):
        reserve_idx = self.reserve_layer
        if reserve_idx < 0:
            reserve_idx = num_layers + reserve_idx
        return reserve_idx

    def clear_latest_routing_decision(self):
        self._latest_routing_decision = None

    def _cache_latest_routing_decision(
        self,
        selection_mode: str,
        layers: torch.Tensor,
        probs: torch.Tensor,
        reserve_layer_prob: Optional[torch.Tensor] = None,
        extra_payload: Optional[dict] = None,
    ) -> None:
        if self.training:
            self._latest_routing_decision = None
            return

        payload = {
            "selection_mode": selection_mode,
            "num_layers_available": int(self.num_layers) if self.num_layers is not None else None,
            "num_layers_to_choose": int(layers.shape[-1]),
            "layers": layers.detach().cpu(),
            "probs": probs.detach().to(torch.float32).cpu(),
            "reserve_layer_index": None,
            "reserve_layer_prob": None,
        }
        if reserve_layer_prob is not None:
            payload["reserve_layer_index"] = int(self._resolve_reserve_idx(self.num_layers))
            payload["reserve_layer_prob"] = reserve_layer_prob.detach().to(torch.float32).cpu()
        if extra_payload is not None:
            for key, value in extra_payload.items():
                if torch.is_tensor(value):
                    payload[key] = value.detach().to(torch.float32).cpu()
                else:
                    payload[key] = value

        self._latest_routing_decision = payload

    def _select_router_tokens(self, router_probs: torch.Tensor) -> torch.Tensor:
        strategy = self.vision_feature_select_strategy
        if strategy in ("default", "cls_patch"):
            return router_probs
        if strategy == "patch":
            return router_probs[:, 1:, :]
        if strategy == "cls":
            return router_probs[:, :1, :]
        raise ValueError(f"Unexpected select feature: {strategy}")

    @staticmethod
    def _cast_tensor_for_module(tensor: torch.Tensor, module: Optional[nn.Module]) -> torch.Tensor:
        if tensor is None or module is None:
            return tensor
        param = next(module.parameters(), None)
        if param is None:
            return tensor
        if tensor.device != param.device or tensor.dtype != param.dtype:
            tensor = tensor.to(device=param.device, dtype=param.dtype)
        return tensor

    def _topk_expert_mask(self, router_probs: torch.Tensor) -> torch.Tensor:
        k = min(self.num_layers_to_choose, router_probs.shape[-1])
        topk_layers = torch.topk(router_probs, k=k, dim=-1).indices
        expert_mask_full = F.one_hot(topk_layers, num_classes=router_probs.shape[-1]).float().sum(dim=1)
        expert_mask_full = F.normalize(expert_mask_full, p=1, dim=-1)
        return expert_mask_full

    def _maybe_save_expert_mask(self, expert_mask_full: torch.Tensor):
        if self.save_router_stats and self.training:
            if self.step_ep % self.save_steps_router_stats or self.step_ep == 0:
                self.expert_mask_per_step[self.step_ep] = (
                    expert_mask_full.cpu().to(torch.float32).detach()
                )
            self.step_ep += 1

    def _load_balance_loss_image_level(self, router_probs: torch.Tensor):
        assert self.num_layers is not None
        expert_mask_full = self._topk_expert_mask(router_probs)
        self._maybe_save_expert_mask(expert_mask_full)

        density_1 = torch.mean(expert_mask_full, dim=0)  # shape: [L]
        density_1_proxy = torch.mean(router_probs, dim=0)  # shape: [L]
        loss = torch.mean(density_1_proxy * density_1) * (self.num_layers**2)
        return loss

    def _load_balance_loss_patch_level(self, aux_payload):
        router_probs = aux_payload["router_probs"]
        assert router_probs.dim() == 3

        # Match the set of vision tokens that survive after routing.
        router_probs = self._select_router_tokens(router_probs)
        router_probs_for_selection = aux_payload.get("router_probs_for_selection")
        reserve_layer_prob = aux_payload.get("reserve_layer_prob")

        if router_probs_for_selection is None or (
            self.force_reserve_layer and reserve_layer_prob is None
        ):
            raise ValueError(
                "patch_layer aux payload must include router_probs_for_selection "
                "and reserve_layer_prob."
            )

        router_probs_for_selection = self._select_router_tokens(router_probs_for_selection)
        if reserve_layer_prob is not None:
            reserve_layer_prob = self._select_router_tokens(
                reserve_layer_prob.unsqueeze(-1)
            ).squeeze(-1)

        if self.force_reserve_layer:
            reserve_idx = self._resolve_reserve_idx(router_probs.shape[-1])
            non_reserved_probs = torch.cat(
                (router_probs[:, :, :reserve_idx], router_probs[:, :, reserve_idx + 1 :]),
                dim=-1,
            )
            selection_probs = torch.cat(
                (
                    router_probs_for_selection[:, :, :reserve_idx],
                    router_probs_for_selection[:, :, reserve_idx + 1 :],
                ),
                dim=-1,
            )
            patch_weight = (1.0 - reserve_layer_prob).clamp(min=0.0)
            soft_proxy = torch.where(
                patch_weight.unsqueeze(-1) > 0,
                non_reserved_probs / patch_weight.unsqueeze(-1).clamp(min=1e-8),
                torch.zeros_like(non_reserved_probs),
            )  # `soft_proxy` re-normalizes the non-reserved layer probs so they sum to 1
        else:
            selection_probs = router_probs_for_selection
            soft_proxy = router_probs
            patch_weight = torch.ones(
                router_probs.shape[:2], device=router_probs.device, dtype=router_probs.dtype
            )

        batch_size, num_tokens, num_layers = selection_probs.shape
        flat_selection_probs = selection_probs.reshape(batch_size * num_tokens, num_layers)
        flat_soft_proxy = soft_proxy.reshape(batch_size * num_tokens, num_layers)
        flat_patch_weight = patch_weight.reshape(batch_size * num_tokens, 1)

        expert_mask_full = self._topk_expert_mask(flat_selection_probs)
        self._maybe_save_expert_mask(expert_mask_full)

        total_weight = flat_patch_weight.sum().clamp(min=1e-8)
        density_1 = torch.sum(expert_mask_full * flat_patch_weight, dim=0) / total_weight
        density_1_proxy = torch.sum(flat_soft_proxy * flat_patch_weight, dim=0) / total_weight
        loss = torch.mean(density_1_proxy * density_1) * (num_layers**2)
        return loss

    def load_balance_loss(self, router_probs):
        """Calculate load−balancing loss to ensure diverse expert routing. Taken from swift transformer pseudo code.
        Removed tensor_cores because we do not need to restrict input number of tokens to any expert """
        if isinstance(router_probs, dict):
            kind = router_probs.get("kind")
            if kind == "patch_layer":
                return self._load_balance_loss_patch_level(router_probs)
            raise ValueError(f"Unsupported auxiliary router payload kind: {kind!r}")
        return self._load_balance_loss_image_level(router_probs)

    def cap_layer_prob(self, router_probs: torch.Tensor, cap: float, layer: int, eps=1e-12):
        """
        router_probs: [B, L] softmax probs
        cap: float, e.g. 0.06, enforce probs[:, layer] <= cap
        Returns: probs' [B, L], sums to 1, preserves ranking among i != layer.
        """
        assert router_probs.dim() == 2
        assert 0.0 <= cap < 1.0
        B, L = router_probs.shape
        if layer == -1 :
            layer = L - 1

        p = router_probs

        p_layer = p[:, layer]
        cap_t = torch.tensor(cap, device=p.device, dtype=p.dtype)

        # rows where we need to cap
        need = p_layer > cap_t
        if not need.any():
            return p

        # sum of non-last mass
        p_other_sum = p[:, :layer].sum(dim=-1) +  p[:, layer + 1:].sum(dim=-1)
        # print(p_other_sum)
        # scale factor to make non-last sum become (1 - cap)
        scale = (1.0 - cap_t) / (p_other_sum + eps)

        # build output
        p_new = p.clone()

        # only apply to rows that need it
        p_new[need, :layer] = p_new[need, :layer] * scale[need].unsqueeze(-1)
        p_new[need, layer + 1 : ] = p_new[need, layer + 1 : ] * scale[need].unsqueeze(-1)
        p_new[need, layer] = cap_t

        s = p_new.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5, rtol=1e-5)

        return p_new
    
    def router_probs(self, hidden_states, text_embeds=None, text_attention_mask=None):
        # hidden states are of size:  torch.Size([num_layers, batch_size, num_tokens, hidden_size]).
        # print("hidden_staes", hidden_states.shape)
        # print("weights: ", self.weights.shape)
        if hidden_states.device != self.weights.device or hidden_states.dtype != self.weights.dtype:
            hidden_states = hidden_states.to(device=self.weights.device, dtype=self.weights.dtype)

        if self.weights.numel() == 0:
            """
            if we here and the zero3 script is zero3 offload, then that means that the self.weights
            is offloaded, resulting in something like : rank 0 weights.shape (0,) weights.device cuda:0 numel 0 requires_grad True
            to fix this problem, we need to gather it from deepspeed
            """
            import deepspeed
            with deepspeed.zero.GatheredParameters([self.weights], modifier_rank=None):
                router_logits = torch.einsum("lbnh, nh -> bl", hidden_states, self.weights)
                print("\n Successfully gathered the weights and computed router_logits")
        else:
            router_logits = torch.einsum("lbnh, nh -> bl", hidden_states, self.weights)

        if self.training:
            # Use the device from hidden_states to avoid issues with offloaded weights
            weights_device = hidden_states.device
            random_noise = torch.rand(router_logits.shape).to(device=weights_device)
            router_logits += random_noise # add random noise to enable exploration to avoid collapse in behavior where always choose same expert
       
        # Convert input to softmax operation from bfloat16 to float32 for stability. <- do I need to do this?
        router_logits = router_logits.to(torch.float32)

        temperature = self.router_temperature if self.router_temperature is not None else 1.0
        if temperature != 1.0:
            router_logits = router_logits / temperature
        
        if self.learnable_gate and self.confidence_gate:
            raise ValueError("learnable_gate and confidence_gate are mutually exclusive.")
        
        use_gate = self.force_reserve_layer and (self.learnable_gate or self.confidence_gate)
        if use_gate:
            # NOTE: calculate softmax **only over non-reserved layers**
            # And assign the reserved layer a separate weight via: (i) a learnable gate or (ii) confidence-based gate.
            reserve_idx = self.reserve_layer
            if reserve_idx < 0:
                reserve_idx = router_logits.shape[-1] + reserve_idx
            # compute softmax over non-reserved layers only
            masked_logits = router_logits.clone()
            masked_logits[:, reserve_idx] = -1e9
            non_reserved_probs = torch.softmax(masked_logits, dim=-1)

            if self.learnable_gate:
                # learn a separate gate for the reserved layer from its own hidden states
                if self.reserve_gate is None:
                    self.reserve_gate = nn.Linear(hidden_states.shape[-1], 1).to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                reserve_hidden = hidden_states[self.reserve_layer]  # [B, N, H]
                pooled = reserve_hidden.mean(dim=1)  # [B, H]
                pooled = self._cast_tensor_for_module(pooled, self.reserve_gate)
                gate_logits = self.reserve_gate(pooled).squeeze(-1)
                reserve_layer_prob = torch.sigmoid(gate_logits)
            else:
                # confidence-based gate: higher confidence in non-reserved -> lower reserved weight.
                conf = non_reserved_probs.max(dim=-1).values
                reserve_layer_prob = torch.sigmoid(
                    self.confidence_gate_slope * (self.confidence_gate_threshold - conf)
                )

            router_probs = non_reserved_probs * (1.0 - reserve_layer_prob).unsqueeze(-1)
            router_probs[:, reserve_idx] = reserve_layer_prob
            router_probs = router_probs.to(router_logits.dtype)
        else:
            router_probs = torch.softmax(router_logits, dim=-1).to(router_logits.dtype)
        
        if self.save_router_stats and self.training:
            # print("self.router_info : ", self.router_info)
            # import sys; sys.exit()
            if self.step_rp % self.save_steps_router_stats or self.step_rp == 0:
                self.router_probs_per_step[self.step_rp] = router_probs.cpu().to(torch.float32).detach() #.numpy().tolist()
            self.step_rp += 1
        # with open("/shared/nas2/sstoica2/MoL/router_probs_per_step.json", 'w') as f:
        #     json.dump(self.router_probs_per_step, f, indent=4)
        # print("router_probs.shape: ", router_probs.shape)
        if self.cap_softmax:
            router_probs = self.cap_layer_prob(router_probs, self.cap_softmax_value, self.reserve_layer)
        return router_probs # torch.Size([batch_size, num_layers])

    def get_layer_and_scaling_probs(self, router_probs):
        # the last dim in router probs is num_layers: 24
        # router_probs = [batch_size, num_layers]
        # exclude the penultimate layer (reserved for last_hidden_state) from top-k selection.

        # TO DO: modify the code to just have router prob choose the k layers.

        router_probs = router_probs.clone()
        if self.force_reserve_layer:
            reserve_layer_prob = router_probs[:, self.reserve_layer]
            # prevent the reserved layer from being selected for top-k,
            # but keep its original probability for scaling.
            router_probs_for_topk = router_probs.clone()
            router_probs_for_topk[:, self.reserve_layer] = -1.0
        else:
            router_probs_for_topk = router_probs

        sorted_probs, sorted_idxs = torch.sort(router_probs_for_topk, dim=-1, descending=True)
        probs = sorted_probs[:, :self.num_layers_to_choose]
        layers = sorted_idxs[:, :self.num_layers_to_choose]
        
        if self.force_reserve_layer:
            # renormalize top-k probs to sum to (1 - reserve_layer_prob)
            remaining_mass = (1.0 - reserve_layer_prob).clamp(min=0.0)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            safe_probs_sum = probs_sum.clamp(min=1e-8)
            scale = torch.where(
                probs_sum > 0,
                remaining_mass.unsqueeze(-1) / safe_probs_sum,
                torch.zeros_like(probs_sum),
            )
            # NOTE: we are just adjusting the scale of probs to sum to (1 - reserve_layer_prob), 
            # so that when we add the reserve_layer_prob later, the total sum is 1.
            probs = probs * scale
            return layers, probs, reserve_layer_prob  # [batch_size, k], [batch_size, k ], [batch_size ]
        
        return layers, probs # [batch_size, k], [batch_size, k  ]

    def forward(self, hidden_states, text_embeds=None, text_attention_mask=None):
        """Produce the combine and dispatch tensors used for sending and
        receiving tokens from their highest probability expert. """
        # inputs shape: [num cores, tokens per core, d model]
        # hidden states are of size:  torch.Size([num_layers (L), batch_size (B), num_tokens (N), hidden_size (H)]). 
        L, B, N, H = hidden_states.shape
        k = self.num_layers_to_choose
        self.num_layers = L
        self.clear_latest_routing_decision()
        router_probs = self.router_probs(hidden_states)
        self._latest_router_probs_for_aux = router_probs
        if self.force_reserve_layer:
            layers, probs, reserve_layer_prob = self.get_layer_and_scaling_probs(router_probs)
            self._cache_latest_routing_decision(
                "image_level",
                layers,
                probs,
                reserve_layer_prob=reserve_layer_prob,
            )
        else:
            layers, probs = self.get_layer_and_scaling_probs(router_probs)
            self._cache_latest_routing_decision("image_level", layers, probs)
        
        # use torch.gather to gather the indexes in layer to get image_features of shape [B, L, N, H]
        hidden_states = hidden_states.transpose(0, 1)
        layers = layers.reshape(B, -1, 1, 1).expand(B, k, N, H) # [B, k, N, H]     
        hidden_states_chosen = torch.gather(hidden_states, 1, layers)  # [B, k, N, H]     
        if self.force_reserve_layer:
            reserve_hidden_state = hidden_states[:, self.reserve_layer, :, :].unsqueeze(1) # [B, 1, N, H]
        # print("hidden_states_chosen: ", hidden_states_chosen.shape)
        # print("last_hidden_state: ", hidden_states_chosen.shape)
        probs = probs.to(hidden_states_chosen.dtype) # convert from float32 back to bfloat16
        image_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        # print("image_interm_features: ", image_interm_features.shape)
        if k > 1:
            image_features = image_features.sum(dim=1, keepdim=True) # need to get [B, 1, N, H]

        if self.force_reserve_layer:
            # print("reserve_layer_prob.reshape(B, -1, 1, 1): ", reserve_layer_prob.reshape(B, -1, 1, 1).shape)
            reserve_layer_prob = reserve_layer_prob.to(hidden_states_chosen.dtype)
            image_features = image_features + reserve_layer_prob.reshape(B, -1, 1, 1) * reserve_hidden_state # [B, 1, N, H]

        image_features = image_features.squeeze(1) # [B, N, H]
        image_features = self._select_vision_features(image_features)
        assert image_features.ndim == 3
        return image_features
    

class TextConditionedRouter(Router):
    """    
    Extension of the Router class above that conditions on input text to decide
    which layers to pick from the vision tower.
    """
    def __init__(
        self,
        config,
        weights,
        top_k=1,
        load_balancing_alpha=0.01,
        text_pool: str = "mean",
        text_scale: float = 1.0, # overall influence of text. Would this allow us to better modulate the features. hyperparam sweep
        text_conditioning: str = "film", # "film" or "attention"
        text_attention_type: str = "concat", # "concat" or "qk"
        text_attention_dim: Optional[int] = None,
        attn_level: str = "layer", # "token", "layer", "patch_layer", or "hybrid" (used when text_conditioning == "attention")
        hybrid_reserve_mode: str = "local_gate",  # the "local_gate" uses only vision hidden states to compute the reserved layer gate, while "disagreement_gate" uses the disagreement between local and global router probs to compute the gate.
        hybrid_disagreement_metric: str = "kl_local_global",
        force_reserve_layer: bool = False,
        reserve_layer: int = -2,
        learnable_gate: bool = False,
        confidence_gate: bool = False,
        confidence_gate_threshold: float = 0.6,
        confidence_gate_slope: float = 10.0,
        vision_feature_select_strategy: str = "cls_patch",
    ):
        super().__init__(
            config,
            weights,
            top_k,
            load_balancing_alpha,
            force_reserve_layer=force_reserve_layer,
            reserve_layer=reserve_layer,
            learnable_gate=learnable_gate,
            confidence_gate=confidence_gate,
            confidence_gate_threshold=confidence_gate_threshold,
            confidence_gate_slope=confidence_gate_slope,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        self.text_pool = text_pool
        self.text_scale = text_scale
        self.text_conditioning = text_conditioning
        self.text_attention_type = text_attention_type
        self.attn_level = attn_level
        self.hybrid_reserve_mode = getattr(config, "hybrid_reserve_mode", hybrid_reserve_mode)
        self.hybrid_disagreement_metric = getattr(
            config,
            "hybrid_disagreement_metric",
            hybrid_disagreement_metric,
        )
        self.hybrid_beta_gate = None
        self.hybrid_reserve_d_weight = None
        self.hybrid_reserve_d_bias = None
        self.vision_hidden_size = self._infer_vision_hidden_size(config, weights)
        if self.vision_hidden_size is None:
            raise ValueError(
                "Unable to infer vision hidden size for TextConditionedRouter. "
                "Pass weights with a valid shape or provide config.mm_hidden_size/"
                "config.vision_config.hidden_size."
            )
        self.text_hidden_size = self._infer_text_hidden_size(config)
        if self.text_hidden_size is None:
            raise ValueError(
                "Unable to infer text hidden size for TextConditionedRouter. "
                "Provide config.mm_text_hidden_size or config.text_config.hidden_size."
            )
        # Conditioning modules.
        if self.text_conditioning not in ("film", "attention", "attn"):
            raise ValueError(f"Unknown text_conditioning: {self.text_conditioning}")
        if self.attn_level not in ("token", "layer", "patch_layer", "hybrid"):
            raise ValueError(f"Unknown attn_level: {self.attn_level}")
        if self.attn_level == "hybrid" and self.text_conditioning == "film":
            raise ValueError("attn_level='hybrid' requires attention-based text conditioning.")
        if self.hybrid_reserve_mode not in ("local_gate", "disagreement_gate"):
            raise ValueError(f"Unknown hybrid_reserve_mode: {self.hybrid_reserve_mode}")
        if self.hybrid_disagreement_metric not in ("kl_local_global",):
            raise ValueError(
                f"Unknown hybrid_disagreement_metric: {self.hybrid_disagreement_metric}"
            )
        if self.hybrid_reserve_mode == "disagreement_gate" and self.attn_level != "hybrid":
            raise ValueError(
                "hybrid_reserve_mode='disagreement_gate' requires attn_level='hybrid'."
            )
        if self.text_conditioning == "film":
            # Map pooled text embedding -> FiLM (gamma, beta) for vision hidden size.
            self.text_to_film = nn.Linear(self.text_hidden_size, self.vision_hidden_size * 2)
            self.text_attn_mlp = None
            self.text_attn_q = None
            self.text_attn_k = None
            self.text_attn_scale = None
        else:
            self.text_to_film = None
            if self.text_attention_type not in ("concat", "qk"):
                raise ValueError(f"Unknown text_attention_type: {self.text_attention_type}")
            if self.text_attention_type == "concat":
                # Compatibility score from concatenated [text || token] features.
                self.text_attn_mlp = nn.Linear(self.text_hidden_size + self.vision_hidden_size, 1)
                self.text_attn_q = None
                self.text_attn_k = None
                self.text_attn_scale = None
            else:
                # Project text as query and vision tokens as keys.
                self.text_attn_dim = text_attention_dim or self.vision_hidden_size
                self.text_attn_q = nn.Linear(self.text_hidden_size, self.text_attn_dim)
                self.text_attn_k = nn.Linear(self.vision_hidden_size, self.text_attn_dim)
                self.text_attn_scale = 1.0 / math.sqrt(self.text_attn_dim)
                self.text_attn_mlp = None

        if self.attn_level == "hybrid":
            self.hybrid_beta_gate = nn.Linear(self.text_hidden_size, 1)
            nn.init.zeros_(self.hybrid_beta_gate.weight)
            hybrid_beta_init = getattr(config, "hybrid_beta_init", None)
            if hybrid_beta_init is None:
                hybrid_beta_bias = -1.5
            else:
                hybrid_beta_init = float(min(max(hybrid_beta_init, 1e-4), 1.0 - 1e-4))
                hybrid_beta_bias = math.log(
                    hybrid_beta_init / (1.0 - hybrid_beta_init)
                )
            nn.init.constant_(self.hybrid_beta_gate.bias, hybrid_beta_bias)
            if self.hybrid_reserve_mode == "disagreement_gate":
                self.hybrid_reserve_d_weight = nn.Parameter(torch.tensor(0.0))
                self.hybrid_reserve_d_bias = nn.Parameter(
                    torch.tensor(math.log(0.20 / 0.80))
                )

    def _infer_vision_hidden_size(self, config, weights):
        if weights is not None and weights.ndim == 2 and weights.numel() > 0:
            return weights.shape[-1]
        for name in ("mm_hidden_size", "vision_hidden_size"):
            value = getattr(config, name, None)
            if value is not None:
                return value
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            value = getattr(vision_config, "hidden_size", None)
            if value is not None:
                return value
        return None

    def _infer_text_hidden_size(self, config):
        for name in ("mm_text_hidden_size", "text_hidden_size"):
            value = getattr(config, name, None)
            if value is not None:
                return value
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            value = getattr(text_config, "hidden_size", None)
            if value is not None:
                return value
        return None

    def _pool_text(self, text_embeds, text_attention_mask):
        # Accept already pooled text embeddings (B, H)
        if text_embeds.ndim == 2:
            return text_embeds
        if text_embeds.ndim != 3:
            raise ValueError(f"Expected text_embeds of shape [B, T, H], got {text_embeds.shape}")

        if self.text_pool == "cls":
            return text_embeds[:, 0, :]

        if text_attention_mask is None:
            return text_embeds.mean(dim=1)

        mask = text_attention_mask.to(dtype=text_embeds.dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        return (text_embeds * mask).sum(dim=1) / denom

    def _router_projection_dtype(self):
        for module_name in ("text_to_film", "text_attn_mlp", "text_attn_q", "text_attn_k", "reserve_gate", "hybrid_beta_gate"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            param = next(module.parameters(), None)
            if param is not None:
                return param.dtype
        return self.weights.dtype

    def _cast_router_projection_inputs(self, hidden_states, text_summary):
        router_dtype = self._router_projection_dtype()
        if hidden_states.dtype != router_dtype:
            hidden_states = hidden_states.to(dtype=router_dtype)
        if text_summary.dtype != router_dtype:
            text_summary = text_summary.to(dtype=router_dtype)
        return hidden_states, text_summary

    def _condition_hidden_states_film(self, hidden_states, text_embeds, text_attention_mask):
        text_summary = self._pool_text(text_embeds, text_attention_mask).to(
            device=hidden_states.device
        )
        hidden_states, text_summary = self._cast_router_projection_inputs(
            hidden_states, text_summary
        )
        film = self.text_to_film(text_summary)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)  # keep modulation bounded
        gamma = gamma * self.text_scale # similar to scale based on what layer we are at
        beta = beta * self.text_scale
        gamma = gamma.unsqueeze(0).unsqueeze(2)
        beta = beta.unsqueeze(0).unsqueeze(2)
        return hidden_states * (1 + gamma) + beta

    def _attention_text_summary(self, hidden_states, text_embeds, text_attention_mask):
        text_summary = self._pool_text(text_embeds, text_attention_mask).to(
            device=hidden_states.device
        )
        hidden_states, text_summary = self._cast_router_projection_inputs(
            hidden_states, text_summary
        )
        if self.text_scale is not None:
            text_summary = text_summary * self.text_scale
        return hidden_states, text_summary

    def _compute_layer_attention_scores(self, hidden_states, text_embeds, text_attention_mask):
        hidden_states, text_summary = self._attention_text_summary(
            hidden_states, text_embeds, text_attention_mask
        )
        layer_summary = hidden_states.mean(dim=2)  # [L, B, H]
        if self.text_attention_type == "concat":
            text_expanded = text_summary.unsqueeze(0).expand(layer_summary.shape[0], -1, -1)
            concat = torch.cat([layer_summary, text_expanded], dim=-1)
            scores = self.text_attn_mlp(concat).squeeze(-1)  # [L, B]
        else:
            q = self.text_attn_q(text_summary)  # [B, D]
            k = self.text_attn_k(layer_summary)  # [L, B, D]
            scores = (k * q.unsqueeze(0)).sum(dim=-1) * self.text_attn_scale  # [L, B]
        return scores

    def _compute_patch_attention_scores(self, hidden_states, text_embeds, text_attention_mask):
        hidden_states, text_summary = self._attention_text_summary(
            hidden_states, text_embeds, text_attention_mask
        )
        if self.text_attention_type == "concat":
            text_expanded = text_summary.unsqueeze(0).unsqueeze(2)
            text_expanded = text_expanded.expand(
                hidden_states.shape[0], -1, hidden_states.shape[2], -1
            )
            concat = torch.cat([hidden_states, text_expanded], dim=-1)
            scores = self.text_attn_mlp(concat).squeeze(-1)  # [L, B, N]
        else:
            q = self.text_attn_q(text_summary)  # [B, D]
            k = self.text_attn_k(hidden_states)  # [L, B, N, D]
            scores = (k * q.unsqueeze(0).unsqueeze(2)).sum(dim=-1) * self.text_attn_scale  # [L, B, N]
        return scores

    def _compute_patch_router_scores(self, hidden_states, text_embeds, text_attention_mask):
        scores = self._compute_patch_attention_scores(
            hidden_states, text_embeds, text_attention_mask
        )
        scores = scores.permute(1, 2, 0).to(torch.float32)  # [B, N, L]
        if self.training:
            random_noise = torch.rand(scores.shape, device=scores.device)
            scores = scores + random_noise

        temperature = self.router_temperature if self.router_temperature is not None else 1.0
        if temperature != 1.0:
            scores = scores / temperature
        return scores

    def _compute_hybrid_beta(self, hidden_states, text_embeds, text_attention_mask):
        hybrid_beta_override = getattr(self.config, "hybrid_beta_override", None)
        if hybrid_beta_override is not None:
            hybrid_beta_override = float(min(max(hybrid_beta_override, 0.0), 1.0))
            return torch.full(
                (hidden_states.shape[1], 1, 1),
                hybrid_beta_override,
                device=hidden_states.device,
                dtype=torch.float32,
            )

        _, text_summary = self._attention_text_summary(
            hidden_states, text_embeds, text_attention_mask
        )
        if self.hybrid_beta_gate is None:
            raise ValueError("hybrid_beta_gate is not initialized. Use attn_level='hybrid'.")
        text_summary = self._cast_tensor_for_module(text_summary, self.hybrid_beta_gate)
        beta_logits = self.hybrid_beta_gate(text_summary).to(torch.float32)  # [B, 1]
        return torch.sigmoid(beta_logits).unsqueeze(1)  # [B, 1, 1]

    def _compute_hybrid_disagreement(
        self,
        global_non_reserved: torch.Tensor,
        local_non_reserved: torch.Tensor,
    ) -> torch.Tensor:
        return self._compute_hybrid_disagreement_terms(
            global_non_reserved, local_non_reserved
        ).sum(dim=-1)

    def _compute_hybrid_disagreement_terms(
        self,
        global_non_reserved: torch.Tensor,
        local_non_reserved: torch.Tensor,
    ) -> torch.Tensor:
        if self.hybrid_disagreement_metric != "kl_local_global":
            raise ValueError(
                f"Unsupported hybrid_disagreement_metric: {self.hybrid_disagreement_metric}"
            )
        eps = 1e-8
        global_log = torch.log(global_non_reserved.clamp(min=eps)).unsqueeze(1)
        local_log = torch.log(local_non_reserved.clamp(min=eps))
        return local_non_reserved * (local_log - global_log)

    def _compute_hybrid_reserve_from_disagreement(
        self, disagreement: torch.Tensor
    ) -> torch.Tensor:
        if self.hybrid_reserve_d_weight is None or self.hybrid_reserve_d_bias is None:
            raise ValueError(
                "hybrid_reserve_d_weight and hybrid_reserve_d_bias are not initialized. "
                "Use hybrid_reserve_mode='disagreement_gate'."
            )
        disagreement = disagreement.to(torch.float32)
        weight = self.hybrid_reserve_d_weight.to(
            device=disagreement.device, dtype=torch.float32
        )
        bias = self.hybrid_reserve_d_bias.to(
            device=disagreement.device, dtype=torch.float32
        )
        return torch.sigmoid(weight * disagreement + bias)

    def _make_disagreement_debug_payload(
        self,
        disagreement: torch.Tensor,
        disagreement_terms: torch.Tensor,
        reserve_layer_prob: Optional[torch.Tensor],
        beta: torch.Tensor,
        num_layers: int,
    ) -> Optional[dict]:
        disagreement = self._select_router_tokens(disagreement.unsqueeze(-1)).squeeze(-1)
        disagreement_terms = self._select_router_tokens(disagreement_terms)
        beta = beta.to(torch.float32)
        if reserve_layer_prob is not None:
            reserve_layer_prob = self._select_router_tokens(
                reserve_layer_prob.unsqueeze(-1)
            ).squeeze(-1)

        flat_disagreement = disagreement.reshape(-1).to(torch.float32)
        flat_disagreement_terms = disagreement_terms.reshape(
            -1, disagreement_terms.shape[-1]
        ).to(torch.float32)
        flat_reserve = None
        finite_mask = torch.isfinite(flat_disagreement)
        finite_mask = finite_mask & torch.isfinite(flat_disagreement_terms).all(dim=-1)

        if reserve_layer_prob is not None:
            flat_reserve = reserve_layer_prob.reshape(-1).to(torch.float32)
            finite_mask = finite_mask & torch.isfinite(flat_reserve)

        if not finite_mask.any():
            return None

        flat_disagreement = flat_disagreement[finite_mask]
        flat_disagreement_terms = flat_disagreement_terms[finite_mask]
        if flat_reserve is not None:
            flat_reserve = flat_reserve[finite_mask]

        if self.force_reserve_layer:
            reserve_idx = self._resolve_reserve_idx(num_layers)
            disagreement_per_layer = flat_disagreement_terms.new_zeros(
                flat_disagreement_terms.shape[0], num_layers
            )
            disagreement_per_layer[:, :reserve_idx] = flat_disagreement_terms[:, :reserve_idx]
            disagreement_per_layer[:, reserve_idx + 1 :] = flat_disagreement_terms[:, reserve_idx:]
        else:
            disagreement_per_layer = flat_disagreement_terms

        payload = {
            "disagreement_mean": flat_disagreement.mean().detach().cpu(),
            "disagreement_per_layer_mean": disagreement_per_layer.mean(dim=0).detach().cpu(),
            "beta_mean": beta.mean().detach().cpu(),
        }

        if flat_reserve is not None:
            payload["reserve_gamma_mean"] = flat_reserve.mean().detach().cpu()

        return payload

    def _remove_reserved_layer(self, probs: torch.Tensor, reserve_idx: int) -> torch.Tensor:
        return torch.cat((probs[..., :reserve_idx], probs[..., reserve_idx + 1 :]), dim=-1)

    def _restore_reserved_layer(
        self,
        non_reserved_probs: torch.Tensor,
        reserve_layer_prob: torch.Tensor,
        num_layers: int,
    ) -> torch.Tensor:
        reserve_idx = self._resolve_reserve_idx(num_layers)
        full_probs = non_reserved_probs.new_zeros(non_reserved_probs.shape[:-1] + (num_layers,))
        full_probs[..., :reserve_idx] = non_reserved_probs[..., :reserve_idx]
        full_probs[..., reserve_idx] = reserve_layer_prob
        full_probs[..., reserve_idx + 1 :] = non_reserved_probs[..., reserve_idx:]
        return full_probs

    def _global_layer_router_probs(self, hidden_states, text_embeds, text_attention_mask):
        scores = self._compute_layer_attention_scores(
            hidden_states, text_embeds, text_attention_mask
        )
        scores = scores.permute(1, 0).to(torch.float32)  # [B, L]
        if self.training:
            scores = scores + torch.rand(scores.shape, device=scores.device)

        temperature = self.router_temperature if self.router_temperature is not None else 1.0
        if temperature != 1.0:
            scores = scores / temperature

        if self.force_reserve_layer:
            reserve_idx = self._resolve_reserve_idx(scores.shape[-1])
            masked_scores = scores.clone()
            masked_scores[:, reserve_idx] = -1e9
            return torch.softmax(masked_scores, dim=-1).to(scores.dtype)

        return torch.softmax(scores, dim=-1).to(scores.dtype)

    def _condition_hidden_states_attention(self, hidden_states, text_embeds, text_attention_mask):
        original_hidden_states = hidden_states
        hidden_states, text_summary = self._attention_text_summary(
            hidden_states, text_embeds, text_attention_mask
        )

        if self.attn_level == "token":
            if self.text_attention_type == "concat":
                # [L, B, N, H_text]
                text_expanded = text_summary.unsqueeze(0).unsqueeze(2)
                text_expanded = text_expanded.expand(
                    hidden_states.shape[0], -1, hidden_states.shape[2], -1
                )
                concat = torch.cat([hidden_states, text_expanded], dim=-1)
                scores = self.text_attn_mlp(concat).squeeze(-1)  # [L, B, N]
            else:
                q = self.text_attn_q(text_summary)  # [B, D]
                k = self.text_attn_k(hidden_states)  # [L, B, N, D]
                scores = (k * q.unsqueeze(0).unsqueeze(2)).sum(dim=-1) * self.text_attn_scale

            attn = torch.softmax(scores, dim=-1).to(
                device=original_hidden_states.device,
                dtype=original_hidden_states.dtype,
            )
            return original_hidden_states * attn.unsqueeze(-1)

        if self.attn_level == "layer":
            # Layer-level attention.
            scores = self._compute_layer_attention_scores(
                hidden_states, text_embeds, text_attention_mask
            )

            # Normalize across layers, then scale each layer's tokens.
            attn = torch.softmax(scores, dim=0).to(
                device=original_hidden_states.device,
                dtype=original_hidden_states.dtype,
            )  # [L, B]
            return original_hidden_states * attn.unsqueeze(-1).unsqueeze(-1)

        raise ValueError(
            f"attn_level={self.attn_level} is not supported by hidden-state conditioning; "
            "use patch_layer or hybrid routing path in forward."
        )

    def _resolve_reserve_idx(self, num_layers):
        reserve_idx = self.reserve_layer
        if reserve_idx < 0:
            reserve_idx = num_layers + reserve_idx
        return reserve_idx

    def _patch_layer_router_probs(
        self,
        hidden_states,
        text_embeds,
        text_attention_mask,
        save_stats: bool = True,
        apply_cap: bool = True,
    ):
        """Compute per-patch distributions over layers.

        Returns:
            router_probs: [B, N, L] layer probabilities for each patch token.
            router_probs_for_selection: [B, N, L] probabilities with reserved layer masked out when enabled.
            reserve_layer_prob: [B, N] reserved-layer mixing weight for each patch.
        """
        scores = self._compute_patch_router_scores(
            hidden_states, text_embeds, text_attention_mask
        )

        if self.learnable_gate and self.confidence_gate:
            raise ValueError("learnable_gate and confidence_gate are mutually exclusive.")

        use_gate = self.force_reserve_layer and (self.learnable_gate or self.confidence_gate)
        if use_gate:
            reserve_idx = self._resolve_reserve_idx(scores.shape[-1])
            masked_scores = scores.clone()
            masked_scores[:, :, reserve_idx] = -1e9
            non_reserved_probs = torch.softmax(masked_scores, dim=-1)

            if self.learnable_gate:  
                if self.reserve_gate is None or self.reserve_gate.in_features != hidden_states.shape[-1]:
                    self.reserve_gate = nn.Linear(hidden_states.shape[-1], 1).to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                reserve_hidden = hidden_states[self.reserve_layer]  # [B, N, H]
                reserve_hidden = self._cast_tensor_for_module(reserve_hidden, self.reserve_gate)
                gate_logits = self.reserve_gate(reserve_hidden).squeeze(-1)
                reserve_layer_prob = torch.sigmoid(gate_logits)  # [B, N]
            else:
                conf = non_reserved_probs.max(dim=-1).values
                reserve_layer_prob = torch.sigmoid(
                    self.confidence_gate_slope * (self.confidence_gate_threshold - conf)
                )

            router_probs = non_reserved_probs * (1.0 - reserve_layer_prob).unsqueeze(-1)
            router_probs[:, :, reserve_idx] = reserve_layer_prob
            router_probs = router_probs.to(scores.dtype)
        else:
            router_probs = torch.softmax(scores, dim=-1).to(scores.dtype)  # [B, N, L]

        if save_stats and self.save_router_stats and self.training:
            router_probs_summary = router_probs.mean(dim=1)  # [B, L]
            if self.step_rp % self.save_steps_router_stats or self.step_rp == 0:
                self.router_probs_per_step[self.step_rp] = (
                    router_probs_summary.cpu().to(torch.float32).detach()
                )
            self.step_rp += 1

        if apply_cap and self.cap_softmax:
            B, N, L = router_probs.shape
            router_probs = self.cap_layer_prob(
                router_probs.reshape(B * N, L),
                self.cap_softmax_value,
                self.reserve_layer,
            ).reshape(B, N, L)

        if self.force_reserve_layer:
            reserve_idx = self._resolve_reserve_idx(router_probs.shape[-1])
            router_probs_for_selection = router_probs.clone()
            router_probs_for_selection[:, :, reserve_idx] = -1.0
            reserve_layer_prob = router_probs[:, :, reserve_idx]
            return router_probs, router_probs_for_selection, reserve_layer_prob

        return router_probs, router_probs, None

    def _hybrid_patch_layer_router_probs(
        self,
        hidden_states,
        text_embeds,
        text_attention_mask,
        save_stats: bool = True,
        apply_cap: bool = True,
    ):
        """Fuse global image-level and local patch-level routing via weighted PoE."""
        global_probs = self._global_layer_router_probs(
            hidden_states, text_embeds, text_attention_mask
        )  # [B, L]
        hybrid_extra_payload = {
            "hybrid_reserve_mode": self.hybrid_reserve_mode,
        }

        if self.hybrid_reserve_mode == "disagreement_gate" and self.force_reserve_layer:
            local_scores = self._compute_patch_router_scores(
                hidden_states, text_embeds, text_attention_mask
            )
            local_probs = torch.softmax(local_scores, dim=-1).to(torch.float32)
            reserve_idx = self._resolve_reserve_idx(local_probs.shape[-1])
            reserve_layer_prob = None
            local_non_reserved_mass = self._remove_reserved_layer(local_probs, reserve_idx)
            remaining_mass = local_probs[:, :, reserve_idx].mul(-1.0).add(1.0).clamp(min=0.0)
            local_non_reserved = torch.where(
                remaining_mass.unsqueeze(-1) > 0,
                local_non_reserved_mass / remaining_mass.unsqueeze(-1).clamp(min=1e-8),
                torch.zeros_like(local_non_reserved_mass),
            )
        else:
            local_probs, _, reserve_layer_prob = self._patch_layer_router_probs(
                hidden_states,
                text_embeds,
                text_attention_mask,
                save_stats=False,
                apply_cap=False,
            )  # [B, N, L]

            if self.force_reserve_layer:
                reserve_idx = self._resolve_reserve_idx(local_probs.shape[-1])
                local_non_reserved_mass = self._remove_reserved_layer(local_probs, reserve_idx)
                remaining_mass = (1.0 - reserve_layer_prob).clamp(min=0.0)
                local_non_reserved = torch.where(
                    remaining_mass.unsqueeze(-1) > 0,
                    local_non_reserved_mass / remaining_mass.unsqueeze(-1).clamp(min=1e-8),
                    torch.zeros_like(local_non_reserved_mass),
                )
            else:
                local_non_reserved = local_probs
        beta = self._compute_hybrid_beta(
            hidden_states, text_embeds, text_attention_mask
        )  # [B, 1, 1]

        if self.force_reserve_layer:
            global_non_reserved = self._remove_reserved_layer(global_probs, reserve_idx)
        else:
            global_non_reserved = global_probs

        eps = 1e-8
        hybrid_logits = (
            (1.0 - beta) * torch.log(global_non_reserved.clamp(min=eps)).unsqueeze(1)
            + beta * torch.log(local_non_reserved.clamp(min=eps))
        )
        hybrid_non_reserved = torch.softmax(hybrid_logits, dim=-1).to(local_probs.dtype)
        disagreement_terms = self._compute_hybrid_disagreement_terms(
            global_non_reserved, local_non_reserved
        )
        disagreement = disagreement_terms.sum(dim=-1)

        if self.force_reserve_layer:
            if self.hybrid_reserve_mode == "disagreement_gate":
                reserve_layer_prob = self._compute_hybrid_reserve_from_disagreement(
                    disagreement
                ).to(local_probs.dtype)
                hybrid_extra_payload["hybrid_disagreement_metric"] = (
                    self.hybrid_disagreement_metric
                )
                hybrid_extra_payload["hybrid_disagreement"] = disagreement
            routed_mass = hybrid_non_reserved * (1.0 - reserve_layer_prob).unsqueeze(-1)
            router_probs = self._restore_reserved_layer(
                routed_mass,
                reserve_layer_prob,
                local_probs.shape[-1],
            )
        else:
            router_probs = hybrid_non_reserved

        if apply_cap and self.cap_softmax:
            B, N, L = router_probs.shape
            router_probs = self.cap_layer_prob(
                router_probs.reshape(B * N, L),
                self.cap_softmax_value,
                self.reserve_layer,
            ).reshape(B, N, L)

        if self.force_reserve_layer:
            reserve_idx = self._resolve_reserve_idx(router_probs.shape[-1])
            router_probs_for_selection = router_probs.clone()
            router_probs_for_selection[:, :, reserve_idx] = -1.0
            reserve_layer_prob = router_probs[:, :, reserve_idx]
        else:
            router_probs_for_selection = router_probs
            reserve_layer_prob = None

        if save_stats and self.save_router_stats and self.training:
            router_probs_summary = router_probs.mean(dim=1)  # [B, L]
            if self.step_rp % self.save_steps_router_stats or self.step_rp == 0:
                self.router_probs_per_step[self.step_rp] = (
                    router_probs_summary.cpu().to(torch.float32).detach()
                )
                disagreement_payload = self._make_disagreement_debug_payload(
                    disagreement,
                    disagreement_terms,
                    reserve_layer_prob,
                    beta,
                    router_probs.shape[-1],
                )
                if disagreement_payload is not None:
                    self.hybrid_debug_per_step[self.step_rp] = disagreement_payload
            self.step_rp += 1

        return router_probs, router_probs_for_selection, reserve_layer_prob, hybrid_extra_payload

    def _get_patch_layer_and_scaling_probs(
        self,
        router_probs: torch.Tensor,
        router_probs_for_selection: torch.Tensor,
        reserve_layer_prob: Optional[torch.Tensor] = None,
    ):
        available_layers = router_probs.shape[-1] - (1 if self.force_reserve_layer else 0)
        k = min(self.num_layers_to_choose, available_layers)
        sorted_probs, sorted_idxs = torch.sort(
            router_probs_for_selection, dim=-1, descending=True
        )
        probs = sorted_probs[:, :, :k]
        layers = sorted_idxs[:, :, :k]

        if self.force_reserve_layer:
            if reserve_layer_prob is None:
                raise ValueError("reserve_layer_prob must be provided when force_reserve_layer=True.")
            remaining_mass = (1.0 - reserve_layer_prob).clamp(min=0.0)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            safe_probs_sum = probs_sum.clamp(min=1e-8)
            scale = torch.where(
                probs_sum > 0,
                remaining_mass.unsqueeze(-1) / safe_probs_sum,
                torch.zeros_like(probs_sum),
            )
            probs = probs * scale

        return layers, probs, reserve_layer_prob

    def _condition_hidden_states(self, hidden_states, text_embeds, text_attention_mask):
        if self.text_conditioning == "film":
            return self._condition_hidden_states_film(hidden_states, text_embeds, text_attention_mask)
        return self._condition_hidden_states_attention(hidden_states, text_embeds, text_attention_mask)

    def router_probs(self, hidden_states, text_embeds=None, text_attention_mask=None):
        if text_embeds is None:
            return super().router_probs(hidden_states)
        if (
            self.text_conditioning in ("attention", "attn")
            and self.attn_level == "patch_layer"
        ):  # patch-level, layer-wise attention conditioning with per-patch routing
            patch_router_probs, _, _ = self._patch_layer_router_probs(
                hidden_states, text_embeds, text_attention_mask
            )
            # Keep API compatible with [B, L] expected by auxiliary loss code.
            return patch_router_probs.mean(dim=1)
        if (
            self.text_conditioning in ("attention", "attn")
            and self.attn_level == "hybrid"
        ):
            hybrid_router_probs, _, _, _ = self._hybrid_patch_layer_router_probs(
                hidden_states, text_embeds, text_attention_mask
            )
            return hybrid_router_probs.mean(dim=1)
        conditioned_hidden_states = self._condition_hidden_states(
            hidden_states, text_embeds, text_attention_mask
        )
        return super().router_probs(conditioned_hidden_states)

    def forward(self, hidden_states, text_embeds=None, text_attention_mask=None):
        """Route vision layers conditioned on text embeddings.

        Args:
            hidden_states: [L, B, N, H] vision hidden states.
            text_embeds: [B, T, H_text] or [B, H_text] text embeddings.
            text_attention_mask: [B, T] attention mask for pooling.
        """
        L, B, N, H = hidden_states.shape
        self.clear_latest_routing_decision()
        if (
            text_embeds is not None
            and self.text_conditioning in ("attention", "attn")
            and self.attn_level == "patch_layer"
        ):  # patch-level attention conditioning with per-patch top-k layer routing
            self.num_layers = L
            router_probs, router_probs_for_selection, reserve_layer_prob = self._patch_layer_router_probs(
                hidden_states, text_embeds, text_attention_mask
            )
            # Cache the per-patch router probabilities for auxiliary loss.
            self._latest_router_probs_for_aux = {
                "kind": "patch_layer",
                "router_probs": router_probs,
                "router_probs_for_selection": router_probs_for_selection,
                "reserve_layer_prob": reserve_layer_prob,
            }

            hidden_states = hidden_states.permute(1, 2, 0, 3)  # [B, N, L, H]
            layers, probs, reserve_layer_prob = self._get_patch_layer_and_scaling_probs(
                router_probs,
                router_probs_for_selection,
                reserve_layer_prob,
            )
            self._cache_latest_routing_decision(
                "patch_level",
                layers,
                probs,
                reserve_layer_prob=reserve_layer_prob,
            )
            gather_idx = layers.unsqueeze(-1).expand(-1, -1, -1, H)
            selected_hidden_states = torch.gather(hidden_states, 2, gather_idx)  # [B, N, k, H]
            probs = probs.to(hidden_states.dtype)
            image_features = (probs.unsqueeze(-1) * selected_hidden_states).sum(dim=2)  # [B, N, H]

            if self.force_reserve_layer:
                reserve_idx = self._resolve_reserve_idx(L)
                reserve_hidden_state = hidden_states[:, :, reserve_idx, :]  # [B, N, H]
                reserve_layer_prob = reserve_layer_prob.to(hidden_states.dtype)
                image_features = image_features + reserve_layer_prob.unsqueeze(-1) * reserve_hidden_state

            image_features = self._select_vision_features(image_features)
            assert image_features.ndim == 3
            return image_features

        if (
            text_embeds is not None
            and self.text_conditioning in ("attention", "attn")
            and self.attn_level == "hybrid"
        ):  # hybrid global-local attention with per-patch top-k layer routing
            self.num_layers = L
            (
                router_probs,
                router_probs_for_selection,
                reserve_layer_prob,
                hybrid_extra_payload,
            ) = self._hybrid_patch_layer_router_probs(
                hidden_states, text_embeds, text_attention_mask
            )
            self._latest_router_probs_for_aux = {
                "kind": "patch_layer",
                "router_probs": router_probs,
                "router_probs_for_selection": router_probs_for_selection,
                "reserve_layer_prob": reserve_layer_prob,
            }

            hidden_states = hidden_states.permute(1, 2, 0, 3)  # [B, N, L, H]
            layers, probs, reserve_layer_prob = self._get_patch_layer_and_scaling_probs(
                router_probs,
                router_probs_for_selection,
                reserve_layer_prob,
            )
            self._cache_latest_routing_decision(
                "hybrid_patch_level",
                layers,
                probs,
                reserve_layer_prob=reserve_layer_prob,
                extra_payload=hybrid_extra_payload,
            )
            gather_idx = layers.unsqueeze(-1).expand(-1, -1, -1, H)
            selected_hidden_states = torch.gather(hidden_states, 2, gather_idx)  # [B, N, k, H]
            probs = probs.to(hidden_states.dtype)
            image_features = (probs.unsqueeze(-1) * selected_hidden_states).sum(dim=2)  # [B, N, H]

            if self.force_reserve_layer:
                reserve_idx = self._resolve_reserve_idx(L)
                reserve_hidden_state = hidden_states[:, :, reserve_idx, :]  # [B, N, H]
                reserve_layer_prob = reserve_layer_prob.to(hidden_states.dtype)
                image_features = image_features + reserve_layer_prob.unsqueeze(-1) * reserve_hidden_state

            image_features = self._select_vision_features(image_features)
            assert image_features.ndim == 3
            return image_features

        # token, layer, or unconditioned routing path with text-conditioned router probabilities (for layer-level conditioning).
        k = self.num_layers_to_choose
        self.num_layers = L
        router_probs = self.router_probs(hidden_states, text_embeds, text_attention_mask)
        self._latest_router_probs_for_aux = router_probs
        if self.force_reserve_layer:
            layers, probs, reserve_layer_prob = self.get_layer_and_scaling_probs(router_probs)
            self._cache_latest_routing_decision(
                "image_level",
                layers,
                probs,
                reserve_layer_prob=reserve_layer_prob,
            )
        else:
            layers, probs = self.get_layer_and_scaling_probs(router_probs)
            self._cache_latest_routing_decision("image_level", layers, probs)

        hidden_states = hidden_states.transpose(0, 1)
        layers = layers.reshape(B, -1, 1, 1).expand(B, k, N, H) # [B, k, N, H]        
        hidden_states_chosen = torch.gather(hidden_states, 1, layers)  # [B, k, N, H]     
        if self.force_reserve_layer:
            reserve_hidden_state = hidden_states[:, self.reserve_layer, :, :].unsqueeze(1) # [B, 1, N, H]
        # print("hidden_states_chosen: ", hidden_states_chosen.shape)
        # print("last_hidden_state: ", hidden_states_chosen.shape)
        probs = probs.to(hidden_states_chosen.dtype) # convert from float32 back to bfloat16
        image_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        # print("image_interm_features: ", image_interm_features.shape)
        if k > 1:
            image_features = image_features.sum(dim=1, keepdim=True) # need to get [B, 1, N, H]

        if self.force_reserve_layer:
            # print("reserve_layer_prob.reshape(B, -1, 1, 1): ", reserve_layer_prob.reshape(B, -1, 1, 1).shape)
            reserve_layer_prob = reserve_layer_prob.to(hidden_states_chosen.dtype)
            image_features = image_features + reserve_layer_prob.reshape(B, -1, 1, 1) * reserve_hidden_state # [B, 1, N, H]

        image_features = image_features.squeeze(1) # [B, N, H]
        image_features = self._select_vision_features(image_features)
        assert image_features.ndim == 3
        return image_features


class TextOnlyRouter(Router):
    """    
    Router that computes layer logits from pooled text embeddings only.
    """
    def __init__(
        self,
        config,
        weights,
        top_k=1,
        load_balancing_alpha=0.01,
        text_pool: str = "mean",
        text_scale: float = 1.0,
        force_reserve_layer: bool = False,
        reserve_layer: int = -2,
        learnable_gate: bool = False,
        confidence_gate: bool = False,
        confidence_gate_threshold: float = 0.6,
        confidence_gate_slope: float = 10.0,
        vision_feature_select_strategy: str = "cls_patch",
    ):
        super().__init__(
            config,
            weights,
            top_k,
            load_balancing_alpha,
            force_reserve_layer=force_reserve_layer,
            reserve_layer=reserve_layer,
            learnable_gate=learnable_gate,
            confidence_gate=confidence_gate,
            confidence_gate_threshold=confidence_gate_threshold,
            confidence_gate_slope=confidence_gate_slope,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        self.text_pool = text_pool
        self.text_scale = text_scale
        self.text_hidden_size = self._infer_text_hidden_size(config)
        if self.text_hidden_size is None:
            raise ValueError(
                "Unable to infer text hidden size for TextOnlyRouter. "
                "Provide config.mm_text_hidden_size or config.text_config.hidden_size."
            )
        self.text_to_router = None
        if self.learnable_gate:
            # Reserve gate uses text summary, not vision hidden states.
            self.reserve_gate = nn.Linear(self.text_hidden_size, 1)
    
    def _maybe_init_text_router(self, num_layers, device=None, dtype=None):
        if self.text_to_router is None or self.text_to_router.out_features != num_layers:
            self.text_to_router = nn.Linear(self.text_hidden_size, num_layers)
            if device is not None or dtype is not None:
                self.text_to_router = self.text_to_router.to(device=device, dtype=dtype)

    def _infer_text_hidden_size(self, config):
        for name in ("mm_text_hidden_size", "text_hidden_size"):
            value = getattr(config, name, None)
            if value is not None:
                return value
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            value = getattr(text_config, "hidden_size", None)
            if value is not None:
                return value
        return None

    def _pool_text(self, text_embeds, text_attention_mask):
        # Accept already pooled text embeddings (B, H)
        if text_embeds.ndim == 2:
            return text_embeds
        if text_embeds.ndim != 3:
            raise ValueError(f"Expected text_embeds of shape [B, T, H], got {text_embeds.shape}")

        if self.text_pool == "cls":
            return text_embeds[:, 0, :]

        if text_attention_mask is None:
            return text_embeds.mean(dim=1)

        mask = text_attention_mask.to(dtype=text_embeds.dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        return (text_embeds * mask).sum(dim=1) / denom

    def router_probs(self, hidden_states, text_embeds=None, text_attention_mask=None):
        if text_embeds is None:
            batch_size = hidden_states.shape[1]
            text_summary = torch.zeros(
                (batch_size, self.text_hidden_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            text_summary = self._pool_text(text_embeds, text_attention_mask).to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )

        if self.text_scale is not None:
            text_summary = text_summary * self.text_scale

        num_layers = hidden_states.shape[0]
        self._maybe_init_text_router(
            num_layers,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        router_logits = self.text_to_router(text_summary)

        if self.training:
            random_noise = torch.rand(router_logits.shape, device=hidden_states.device)
            router_logits += random_noise

        router_logits = router_logits.to(torch.float32)

        temperature = self.router_temperature if self.router_temperature is not None else 1.0
        if temperature != 1.0:
            router_logits = router_logits / temperature

        if self.learnable_gate and self.confidence_gate:
            raise ValueError("learnable_gate and confidence_gate are mutually exclusive.")

        use_gate = self.force_reserve_layer and (self.learnable_gate or self.confidence_gate)
        if use_gate:
            reserve_idx = self.reserve_layer
            if reserve_idx < 0:
                reserve_idx = router_logits.shape[-1] + reserve_idx
            masked_logits = router_logits.clone()
            masked_logits[:, reserve_idx] = -1e9
            non_reserved_probs = torch.softmax(masked_logits, dim=-1)

            if self.learnable_gate:
                if self.reserve_gate is None or self.reserve_gate.in_features != self.text_hidden_size:
                    self.reserve_gate = nn.Linear(self.text_hidden_size, 1).to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                gate_logits = self.reserve_gate(text_summary).squeeze(-1)
                reserve_layer_prob = torch.sigmoid(gate_logits)
            else:
                conf = non_reserved_probs.max(dim=-1).values
                reserve_layer_prob = torch.sigmoid(
                    self.confidence_gate_slope * (self.confidence_gate_threshold - conf)
                )

            router_probs = non_reserved_probs * (1.0 - reserve_layer_prob).unsqueeze(-1)
            router_probs[:, reserve_idx] = reserve_layer_prob
            router_probs = router_probs.to(router_logits.dtype)
        else:
            router_probs = torch.softmax(router_logits, dim=-1).to(router_logits.dtype)

        if self.save_router_stats and self.training:
            if self.step_rp % self.save_steps_router_stats or self.step_rp == 0:
                self.router_probs_per_step[self.step_rp] = router_probs.cpu().to(torch.float32).detach()
            self.step_rp += 1

        if self.cap_softmax:
            router_probs = self.cap_layer_prob(router_probs, self.cap_softmax_value, self.reserve_layer)
        return router_probs

    def forward(self, hidden_states, text_embeds=None, text_attention_mask=None):
        """Route vision layers using pooled text embeddings only.

        Args:
            hidden_states: [L, B, N, H] vision hidden states.
            text_embeds: [B, T, H_text] or [B, H_text] text embeddings.
            text_attention_mask: [B, T] attention mask for pooling.
        """
        L, B, N, H = hidden_states.shape
        k = self.num_layers_to_choose
        self.num_layers = L
        self.clear_latest_routing_decision()
        router_probs = self.router_probs(hidden_states, text_embeds, text_attention_mask)
        self._latest_router_probs_for_aux = router_probs
        if self.force_reserve_layer:
            layers, probs, reserve_layer_prob = self.get_layer_and_scaling_probs(router_probs)
            self._cache_latest_routing_decision(
                "image_level",
                layers,
                probs,
                reserve_layer_prob=reserve_layer_prob,
            )
        else:
            layers, probs = self.get_layer_and_scaling_probs(router_probs)
            self._cache_latest_routing_decision("image_level", layers, probs)

        hidden_states = hidden_states.transpose(0, 1)
        layers = layers.reshape(B, -1, 1, 1).expand(B, k, N, H) # [B, k, N, H]
        hidden_states_chosen = torch.gather(hidden_states, 1, layers)  # [B, k, N, H]
        if self.force_reserve_layer:
            reserve_hidden_state = hidden_states[:, self.reserve_layer, :, :].unsqueeze(1) # [B, 1, N, H]

        probs = probs.to(hidden_states_chosen.dtype)
        image_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        if k > 1:
            image_features = image_features.sum(dim=1, keepdim=True)

        if self.force_reserve_layer:
            reserve_layer_prob = reserve_layer_prob.to(hidden_states_chosen.dtype)
            image_features = image_features + reserve_layer_prob.reshape(B, -1, 1, 1) * reserve_hidden_state

        image_features = image_features.squeeze(1)
        image_features = self._select_vision_features(image_features)
        assert image_features.ndim == 3
        return image_features
        
