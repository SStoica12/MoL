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
                confidence_gate_threshold=0.6, confidence_gate_slope=10.0):
        super().__init__()
        self.num_layers_to_choose = top_k
        self.config = config
        self.num_layers = None
        self.vision_feature_select_strategy = "default"  # "default", "cls", "patch"
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

    def load_balance_loss(self, router_probs):
        """Calculate load−balancing loss to ensure diverse expert routing. Taken from swift transformer pseudo code.
        Removed tensor_cores because we do not need to restrict input number of tokens to any expert """
        # router probs is the probability assigned for each expert per token.
        """Calculate load−balancing loss to ensure diverse expert routing. Taken from swift transformer pseudo code
        modified to hard code num_experts because we know it is 24, but can find it dynamically too easily.
        Also, removed tensor_cores because we do not need to restrict input number of tokens to any expert 
        
        Args:
            router_probs: [batch_size, num_layers] - probability assigned to each expert per image
        """
        # router_probs = [batch_size, num_layers]
        assert self.num_layers is not None
        # Build a top-k mask over ALL layers (including the reserved layer).
        k = min(self.num_layers_to_choose, self.num_layers)
        topk_layers = torch.topk(router_probs, k=k, dim=-1).indices  # [B, k]
        expert_mask_full = F.one_hot(topk_layers, num_classes=self.num_layers).float().sum(dim=1)
        expert_mask_full = F.normalize(expert_mask_full, p=1, dim=-1)
        
        if self.save_router_stats and self.training:
            # assert self.step_ep not in self.expert_mask_per_step
            if self.step_ep % self.save_steps_router_stats or self.step_ep == 0:
                self.expert_mask_per_step[self.step_ep] = expert_mask_full.cpu().to(torch.float32).detach() #.numpy().tolist()
            self.step_ep += 1
            # with open("./expert_mask_per_step.json", 'w') as f:
            #     json.dump(self.expert_mask_per_step, f, indent=4)
        
        density_1 = torch.mean(expert_mask_full, dim=0)  # shape: [L]
        # For each core, get fraction of probability mass assigned to each expert
        # from the router across all tokens.
        # density 1 proxy shape: [num cores, num experts]
        density_1_proxy = torch.mean(router_probs, dim=0)  # shape: [L]
        # density l for a single core: vector of length num experts that sums to 1.
        # density l proxy for a single core: vector of length num experts that sums to 1.
        # Want both vectors to have uniform allocation (1/num experts) across all num expert elements.
        # The two vectors will be pushed towards uniform allocation when the dot product is minimized.
        loss = torch.mean(density_1_proxy * density_1) * (self.num_layers**2)
        return loss # add to graph so remove .item()

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

        p_layer = p[:, layer]                       # [B]
        cap_t = torch.tensor(cap, device=p.device, dtype=p.dtype)

        # rows where we need to cap
        need = p_layer > cap_t                     # [B]
        if not need.any():
            return p

        # sum of non-last mass
        p_other_sum = p[:, :layer].sum(dim=-1) +  p[:, layer + 1:].sum(dim=-1)    # [B]
        # print(p_other_sum)
        # scale factor to make non-last sum become (1 - cap)
        scale = (1.0 - cap_t) / (p_other_sum + eps)   # [B]

        # build output
        p_new = p.clone()

        # only apply to rows that need it
        p_new[need, :layer] = p_new[need, :layer] * scale[need].unsqueeze(-1)
        p_new[need, layer + 1 : ] = p_new[need, layer + 1 : ] * scale[need].unsqueeze(-1)
        p_new[need, layer] = cap_t

        s = p_new.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5, rtol=1e-5)

        return p_new
    
    def router_probs(self, hidden_states):
        # hidden states are of size:  torch.Size([num_layers, batch_size, num_tokens, hidden_size]).
        # print("hidden_staes", hidden_states.shape)
        # print("weights: ", self.weights.shape)

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
            scale = torch.where(
                probs_sum > 0,
                remaining_mass.unsqueeze(-1) / probs_sum,
                torch.zeros_like(probs_sum),
            )
            # NOTE: we are just adjusting the scale of probs to sum to (1 - reserve_layer_prob), 
            # so that when we add the reserve_layer_prob later, the total sum is 1.
            probs = probs * scale
            return layers, probs, reserve_layer_prob  # [batch_size, k], [batch_size, k ], [batch_size ]
        
        return layers, probs # [batch_size, k], [batch_size, k  ]

    def forward(self, hidden_states):
        """Produce the combine and dispatch tensors used for sending and
        receiving tokens from their highest probability expert. """
        # inputs shape: [num cores, tokens per core, d model]
        # hidden states are of size:  torch.Size([num_layers (L), batch_size (B), num_tokens (N), hidden_size (H)]). 
        L, B, N, H = hidden_states.shape
        k = self.num_layers_to_choose
        self.num_layers = L
        router_probs = self.router_probs(hidden_states)
        if self.force_reserve_layer:
            layers, probs, reserve_layer_prob = self.get_layer_and_scaling_probs(router_probs)
        else:
            layers, probs = self.get_layer_and_scaling_probs(router_probs)
        
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
        if self.vision_feature_select_strategy == "patch": 
            image_features = image_features[:, 1:, :]
        if self.vision_feature_select_strategy == "cls": 
            image_features = image_features[:, 0, :]
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
        attn_level: str = "layer", # "token" or "layer" (used when text_conditioning == "attention")
        force_reserve_layer: bool = False,
        reserve_layer: int = -2,
        learnable_gate: bool = False,
        confidence_gate: bool = False,
        confidence_gate_threshold: float = 0.6,
        confidence_gate_slope: float = 10.0,
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
        )
        self.text_pool = text_pool
        self.text_scale = text_scale
        self.text_conditioning = text_conditioning
        self.text_attention_type = text_attention_type
        self.attn_level = attn_level
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
        if self.attn_level not in ("token", "layer"):
            raise ValueError(f"Unknown attn_level: {self.attn_level}")
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

    def _condition_hidden_states_film(self, hidden_states, text_embeds, text_attention_mask):
        text_summary = self._pool_text(text_embeds, text_attention_mask).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        film = self.text_to_film(text_summary)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)  # keep modulation bounded
        gamma = gamma * self.text_scale # similar to scale based on what layer we are at
        beta = beta * self.text_scale
        gamma = gamma.unsqueeze(0).unsqueeze(2)
        beta = beta.unsqueeze(0).unsqueeze(2)
        return hidden_states * (1 + gamma) + beta

    def _condition_hidden_states_attention(self, hidden_states, text_embeds, text_attention_mask):
        text_summary = self._pool_text(text_embeds, text_attention_mask).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        if self.text_scale is not None:
            text_summary = text_summary * self.text_scale

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

            attn = torch.softmax(scores, dim=-1).to(dtype=hidden_states.dtype)
            return hidden_states * attn.unsqueeze(-1)

        # Layer-level attention.
        layer_summary = hidden_states.mean(dim=2)  # [L, B, H]
        if self.text_attention_type == "concat":
            text_expanded = text_summary.unsqueeze(0).expand(layer_summary.shape[0], -1, -1)
            concat = torch.cat([layer_summary, text_expanded], dim=-1)
            scores = self.text_attn_mlp(concat).squeeze(-1)  # [L, B]
        else:
            q = self.text_attn_q(text_summary)  # [B, D]
            k = self.text_attn_k(layer_summary)  # [L, B, D]
            scores = (k * q.unsqueeze(0)).sum(dim=-1) * self.text_attn_scale  # [L, B]

        # Normalize across layers, then scale each layer's tokens.
        attn = torch.softmax(scores, dim=0).to(dtype=hidden_states.dtype)  # [L, B]
        return hidden_states * attn.unsqueeze(-1).unsqueeze(-1)

    def _condition_hidden_states(self, hidden_states, text_embeds, text_attention_mask):
        if self.text_conditioning == "film":
            return self._condition_hidden_states_film(hidden_states, text_embeds, text_attention_mask)
        return self._condition_hidden_states_attention(hidden_states, text_embeds, text_attention_mask)

    def router_probs(self, hidden_states, text_embeds=None, text_attention_mask=None):
        if text_embeds is None:
            return super().router_probs(hidden_states)
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
        k = self.num_layers_to_choose
        self.num_layers = L
        router_probs = self.router_probs(hidden_states, text_embeds, text_attention_mask)
        if self.force_reserve_layer:
            layers, probs, reserve_layer_prob = self.get_layer_and_scaling_probs(router_probs)
        else:
            layers, probs = self.get_layer_and_scaling_probs(router_probs)

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
        if self.vision_feature_select_strategy == "patch": 
            image_features = image_features[:, 1:, :]
        if self.vision_feature_select_strategy == "cls": 
            image_features = image_features[:, 0, :]
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
        router_probs = self.router_probs(hidden_states, text_embeds, text_attention_mask)
        if self.force_reserve_layer:
            layers, probs, reserve_layer_prob = self.get_layer_and_scaling_probs(router_probs)
        else:
            layers, probs = self.get_layer_and_scaling_probs(router_probs)

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
        if self.vision_feature_select_strategy == "patch": 
            image_features = image_features[:, 1:, :]
        if self.vision_feature_select_strategy == "cls": 
            image_features = image_features[:, 0, :]
        assert image_features.ndim == 3
        return image_features
        
