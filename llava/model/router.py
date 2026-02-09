import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import json

class Router(nn.Module):
    def __init__(self, config, weights, top_k = 1, 
                load_balancing_alpha=0.01, force_reserve_layer=False, 
                reserve_layer=-2, save_router_stats=True, save_steps_router_stats=200, 
                cap_softmax=False, cap_softmax_value=0.06):
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
            #router_probs[:, self.reserve_layer] = -1.0  # prevent the reserved layer from being selected for calculation of top-k
            router_probs = self.cap_layer_prob(router_probs, 0, self.reserve_layer)
        sorted_probs, sorted_idxs = torch.sort(router_probs, dim=-1, descending=True)
        probs = sorted_probs[:, :self.num_layers_to_choose]
        layers = sorted_idxs[:, :self.num_layers_to_choose]
        
        if self.force_reserve_layer:
            reserve_layer_prob = router_probs[:, self.reserve_layer]
            return layers, probs, reserve_layer_prob
        
        # build expert mask (top-k safe): [B, k, L] -> [B, L]
        # NOTE: we don't use this expert_mask anymore
        # expert_mask = F.one_hot(layers, num_classes=self.num_layers).float().sum(dim=1)
        # # normalize so each row sums to 1 (important for top-k > 1) -> routing mass needs to be distributed to sum up to 1
        # expert_mask = F.normalize(expert_mask, p=1, dim=-1)
        
        return layers, probs # [batch_size, num_layers_to_choose] x2, [batch_size, ], [batch_size, num_layers]

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
    
    
    TODO: This is still a work in progress - @JK
    
    
    
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
    ):
        super().__init__(config, weights, top_k, load_balancing_alpha)
        self.text_pool = text_pool
        self.text_scale = text_scale
        self.vision_hidden_size = self._infer_vision_hidden_size(config, weights)
        if self.vision_hidden_size is None:
            raise ValueError(
                "Unable to infer vision hidden size for TextConditionedRouter. "
                "Pass weights with a valid shape or provide config.mm_hidden_size/"
                "config.vision_config.hidden_size."
            )
        # Map pooled text embedding -> FiLM (gamma, beta) for vision hidden size.
        self.text_to_film = nn.LazyLinear(self.vision_hidden_size * 2)  # TODO: check the hidden_state size from text tower

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

    def _condition_hidden_states(self, hidden_states, text_embeds, text_attention_mask):
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
        
        
        


# NOTE: Same logic as when self.num_layers == 23
# class LayerAveragingRouter(nn.Module):
#     '''
#     Weighted averaging across layers with learnable weights.
    
#     '''
    
#     def __init__(self, num_layers: int, vision_feature_select_strategy: str = "default"):
#         super().__init__()
#         if num_layers < 2:
#             raise ValueError("num_layers must be >= 2 to learn coefficients for layers 1..L-1.")
#         self.num_layers = num_layers
#         self.vision_feature_select_strategy = vision_feature_select_strategy
#         self.layer_weights = nn.Parameter(torch.ones(num_layers - 1))

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         # hidden_states: [L, B, N, H]
#         L, B, N, H = hidden_states.shape
#         if L != self.num_layers:
#             raise ValueError(f"Expected {self.num_layers} layers, got {L}.")

#         coeffs = torch.cat([hidden_states.new_ones(1), self.layer_weights.to(hidden_states.dtype)])
#         coeffs = coeffs.view(L, 1, 1, 1)
#         image_feature = (hidden_states * coeffs).sum(dim=0) / L  # layerwise normalization

#         if self.vision_feature_select_strategy == "patch":
#             image_feature = image_feature[:, 1:, :]
#         if self.vision_feature_select_strategy == "cls":
#             image_feature = image_feature[:, 0, :]
#         assert image_feature.ndim == 3
#         return image_feature
