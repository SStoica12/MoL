import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Router(nn.Module):
    def __init__(self, config, weights, top_k = 1, load_balancing_alpha=0.01):
        super().__init__()
        self.num_layers_to_choose = top_k
        self.config = config
        self.num_layers = None
        self.vision_feature_select_strategy = "default"  # "default", "cls", "patch"
        self.weights = nn.Parameter(weights) #[num_tokens, hidden_size]
        self.load_balancing_alpha = load_balancing_alpha
        self.router_temperature = 1.0
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
        #router logits = mtf.to float32(router logits)

        temperature = self.router_temperature if self.router_temperature is not None else 1.0
        if temperature != 1.0:
            router_logits = router_logits / temperature
        router_probs = torch.softmax(router_logits, dim=-1).to(router_logits.dtype)
        # print("router_probs.shape: ", router_probs.shape)
        return router_probs # torch.Size([batch_size, num_layers])

    def get_layer_and_scaling_probs_and_expert_mask(self, router_probs):
        # the last dim in router probs is num_layers: 24
        # router_probs = [batch_size, num_layers]
        # exclude the penultimate layer (reserved for last_hidden_state) from top-k selection.
        router_probs_excluded = router_probs.clone()
        router_probs_excluded[:, -2] = -1.0  # prevent the reserved layer from being selected for calculation of top-k
        sorted_probs, sorted_idxs = torch.sort(router_probs_excluded, dim=-1, descending=True)
        probs = sorted_probs[:, :self.num_layers_to_choose]
        layers = sorted_idxs[:, :self.num_layers_to_choose]
        last_layer_prob = router_probs[:, -2]
        # build expert mask (top-k safe): [B, k, L] -> [B, L]
        # NOTE: we don't use this expert_mask anymore
        expert_mask = F.one_hot(layers, num_classes=self.num_layers).float().sum(dim=1)
        # normalize so each row sums to 1 (important for top-k > 1) -> routing mass needs to be distributed to sum up to 1
        expert_mask = F.normalize(expert_mask, p=1, dim=-1)
        return layers, probs, last_layer_prob, expert_mask # [batch_size, num_layers_to_choose] x2, [batch_size, ], [batch_size, num_layers]

    def forward(self, hidden_states):
        """Produce the combine and dispatch tensors used for sending and
        receiving tokens from their highest probability expert. """
        # inputs shape: [num cores, tokens per core, d model]
        # hidden states are of size:  torch.Size([num_layers (L), batch_size (B), num_tokens (N), hidden_size (H)]). 
        L, B, N, H = hidden_states.shape
        k = self.num_layers_to_choose
        self.num_layers = L
        router_probs = self.router_probs(hidden_states)
        layers, probs, last_layer_prob, _ = self.get_layer_and_scaling_probs_and_expert_mask(router_probs)

        # use torch.gather to gather the indexes in layer to get image_features of shape [B, L, N, H]
        hidden_states = hidden_states.transpose(0, 1)
        layers = layers.reshape(B, -1, 1, 1).expand(B, k, N, H) # [B, k, N, H]     
        hidden_states_chosen = torch.gather(hidden_states, 1, layers)  # [B, k, N, H]     
        last_hidden_state = hidden_states[:, -2, :, :].unsqueeze(1) # [B, 1, N, H]
        # print("hidden_states_chosen: ", hidden_states_chosen.shape)
        # print("last_hidden_state: ", hidden_states_chosen.shape)

        image_interm_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        # print("image_interm_features: ", image_interm_features.shape)
        if k > 1:
            image_interm_features = image_interm_features.sum(dim=1, keepdim=True) # need to get [B, 1, N, H]
        # print("last_layer_prob.reshape(B, -1, 1, 1): ", last_layer_prob.reshape(B, -1, 1, 1).shape)
        image_features = image_interm_features + last_layer_prob.reshape(B, -1, 1, 1) * last_hidden_state

        # print("image feature after aggregation: ", image_features.shape) # [B, 1, N, H]
        image_feature = image_features.squeeze(1) # [B, N, H]
        if self.vision_feature_select_strategy == "patch": 
            image_feature = image_feature[:, 1:, :]
        if self.vision_feature_select_strategy == "cls": 
            image_feature = image_feature[:, 0, :]
        assert image_feature.ndim == 3
        return image_feature
    

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
        text_scale: float = 1.0,
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
        gamma = gamma * self.text_scale
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
        layers, probs, last_layer_prob, _ = self.get_layer_and_scaling_probs_and_expert_mask(router_probs)

        hidden_states_t = hidden_states.transpose(0, 1)  # [B, L, N, H]
        layers_expanded = layers.reshape(B, -1, 1, 1).expand(B, k, N, H)
        hidden_states_chosen = torch.gather(hidden_states_t, 1, layers_expanded)
        last_hidden_state = hidden_states_t[:, -2, :, :].unsqueeze(1)

        image_interm_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        if k > 1:
            image_interm_features = image_interm_features.sum(dim=1, keepdim=True)
        image_features = image_interm_features + last_layer_prob.reshape(B, -1, 1, 1) * last_hidden_state

        image_feature = image_features.squeeze(1)
        if self.vision_feature_select_strategy == "patch":
            image_feature = image_feature[:, 1:, :]
        if self.vision_feature_select_strategy == "cls":
            image_feature = image_feature[:, 0, :]
        assert image_feature.ndim == 3
        return image_feature
        
        
        


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
