import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import json

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

class _RouterWeightModule(nn.Module):
    """Wraps the router weight as its own nn.Module so that DeepSpeed ZeRO-3
    attaches its gather/scatter hooks via __call__ for BOTH forward and backward.

    Without this wrapper, router_probs() is called directly (not via the Router's
    forward()), so ZeRO-3 never gathers self.weights during the backward pass,
    causing a shape mismatch: tensor(0,) vs tensor(1024,).
    """

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = nn.Parameter(weights)  # [num_tokens, hidden_size]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [L, B, N, H], weights: [N, H] -> [B, L]
        return torch.einsum("lbnh, nh -> bl", hidden_states, self.weights) # not applicable for patch_layer, so do not use


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
        rank0_print(f"Initialized Router with num_layers_to_choose={self.num_layers_to_choose}, ")
        self.config = config
        self.num_layers = None
        self.vision_feature_select_strategy = vision_feature_select_strategy  # "patch", "cls_patch", "cls"
        self.layers_clip = None
        self.layers_dino = None
        # _weight_fn is a dedicated sub-module so ZeRO-3 gathers its parameter
        # via __call__ hooks in both forward and backward, even when router_probs()
        # is called directly rather than through Router.forward().
        self._weight_fn = _RouterWeightModule(weights)
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
        self._latest_router_probs_for_aux_clip = None
        self._latest_router_probs_for_aux_dino = None
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

    @property
    def weights(self):
        """Expose the parameter for checkpointing/printing compatibility."""
        return self._weight_fn.weights

    def _select_router_tokens(self, router_probs: torch.Tensor) -> torch.Tensor:
        strategy = self.vision_feature_select_strategy
        if strategy in ("default", "cls_patch"):
            return router_probs
        if strategy == "patch":
            return router_probs[:, 1:, :]
        if strategy == "cls":
            return router_probs[:, :1, :]
        raise ValueError(f"Unexpected select feature: {strategy}")
    
    def _resolve_reserve_idx(self, num_layers):
        reserve_idx = self.reserve_layer
        if reserve_idx < 0:
            reserve_idx = num_layers + reserve_idx
        return reserve_idx
    
    def _select_vision_features(self, image_features: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        strategy = self.vision_feature_select_strategy # cls_patch 
        if strategy in ("default", "cls_patch"):
            return image_features
        if strategy == "patch":
            return image_features[:, 1:, :]
        if strategy == "cls":
            return image_features[:, :1, :]
        raise ValueError(f"Unexpected select feature: {strategy}")
    
    def _topk_expert_mask(self, router_probs: torch.Tensor) -> torch.Tensor: # used in load ballance loss. Follows switch transformer
        k = min(self.num_layers_to_choose, router_probs.shape[-1])
        topk_layers = torch.topk(router_probs, k=k, dim=-1).indices
        expert_mask_full = F.one_hot(topk_layers, num_classes=router_probs.shape[-1]).float().sum(dim=1)
        expert_mask_full = F.normalize(expert_mask_full, p=1, dim=-1)
        return expert_mask_full
    
    def _maybe_save_expert_mask(self, expert_mask_full: torch.Tensor): # for logging during training callback
        if self.save_router_stats and self.training:
            if self.step_ep % self.save_steps_router_stats or self.step_ep == 0:
                self.expert_mask_per_step[self.step_ep] = (
                    expert_mask_full.cpu().to(torch.float32).detach()
                )
            self.step_ep += 1
    
    def _load_balance_loss_image_level(self, router_probs: torch.Tensor): # original load balance loss from switch
        assert self.num_layers is not None
        expert_mask_full = self._topk_expert_mask(router_probs)
        self._maybe_save_expert_mask(expert_mask_full)

        density_1 = torch.mean(expert_mask_full, dim=0)  # shape: [L]
        density_1_proxy = torch.mean(router_probs, dim=0)  # shape: [L]
        loss = torch.mean(density_1_proxy * density_1) * (self.num_layers**2)
        return loss
    
    def _load_balance_loss_patch_level(self, aux_payload):
        # need to define a different load_balance loss because other
        # does per layer. We need per patch
        # rank0_print("aux_payload : ", aux_payload)
        router_probs = aux_payload["router_probs"]
        assert router_probs.dim() == 3

        # Match the set of vision tokens that survive after routing.
        router_probs = self._select_router_tokens(router_probs)
        rank0_print(f"\n ======== router_probs.min() {router_probs.min().item()} and max(): ", router_probs.max().item())
        router_probs_for_selection = aux_payload.get("router_probs_for_selection")
        rank0_print(f"router_probs_for_selection.min() {router_probs_for_selection.min().item()} and max(): ", router_probs_for_selection.max().item())
        reserve_layer_prob = aux_payload.get("reserve_layer_prob")
        rank0_print(f"reserve_layer_prob.min() {reserve_layer_prob.min().item()} and max(): ", reserve_layer_prob.max().item())

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
            rank0_print(f"after selecting vision features, reserve_layer_prob.min() {reserve_layer_prob.min().item()} and max(): ", reserve_layer_prob.max().item())

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
        rank0_print(f"selection_probs.min() {selection_probs.min().item()} and max(): ", selection_probs.max().item())
        rank0_print(f"soft_proxy.min() {soft_proxy.min().item()} and max(): ", soft_proxy.max().item())
        rank0_print(f"patch_weight.min() {patch_weight.min().item()} and max(): ", patch_weight.max().item())
        batch_size, num_tokens, num_layers = selection_probs.shape
        flat_selection_probs = selection_probs.reshape(batch_size * num_tokens, num_layers)
        flat_soft_proxy = soft_proxy.reshape(batch_size * num_tokens, num_layers)
        flat_patch_weight = patch_weight.reshape(batch_size * num_tokens, 1)

        expert_mask_full = self._topk_expert_mask(flat_selection_probs)
        self._maybe_save_expert_mask(expert_mask_full)

        total_weight = flat_patch_weight.sum().clamp(min=1e-8)
        density_1 = torch.sum(expert_mask_full * flat_patch_weight, dim=0) / total_weight
        density_1_proxy = torch.sum(flat_soft_proxy * flat_patch_weight, dim=0) / total_weight
        rank0_print(f"density_1.min() {density_1.min().item()} and max(): ", density_1.max().item())
        rank0_print(f"density_1_proxy.min() {density_1_proxy.min().item()} and max(): ", density_1_proxy.max().item())
        rank0_print(f"=================================== END COMPUTE LOAD BALANCE LOSS ===================================\n")
        loss = torch.mean(density_1_proxy * density_1) * (num_layers**2)
        return loss
    
    def load_balance_loss(self, router_probs):
        """Calculate load−balancing loss to ensure diverse expert routing. Taken from swift transformer pseudo code.
        Removed tensor_cores because we do not need to restrict input number of tokens to any expert """
        if isinstance(router_probs, dict):
            kind = router_probs.get("kind")
            if kind == "patch_layer":
                rank0_print("\n calculating patch layer load balancing loss...")
                return self._load_balance_loss_patch_level(router_probs)
            raise ValueError(f"Unsupported auxiliary router payload kind: {kind!r}")
        return self._load_balance_loss_image_level(router_probs)

    # def load_balance_loss(self, router_probs):
    #     """Calculate load−balancing loss to ensure diverse expert routing. Taken from swift transformer pseudo code.
    #     Removed tensor_cores because we do not need to restrict input number of tokens to any expert """
    #     # router probs is the probability assigned for each expert per token.
    #     """Calculate load−balancing loss to ensure diverse expert routing. Taken from swift transformer pseudo code
    #     modified to hard code num_experts because we know it is 24, but can find it dynamically too easily.
    #     Also, removed tensor_cores because we do not need to restrict input number of tokens to any expert 
        
    #     Args:
    #         router_probs: [batch_size, num_layers] - probability assigned to each expert per image
    #     """
    #     # router_probs = [batch_size, num_layers]
    #     if self.num_layers is None:
    #         self.num_layers = router_probs.shape[1]
    #     assert self.num_layers_to_choose is not None
    #     # Build a top-k mask over ALL layers (including the reserved layer).
    #     k = min(self.num_layers_to_choose, self.num_layers)
    #     topk_layers = torch.topk(router_probs, k=k, dim=-1).indices  # [B, k]
    #     expert_mask_full = F.one_hot(topk_layers, num_classes=self.num_layers).float().sum(dim=1)
    #     expert_mask_full = F.normalize(expert_mask_full, p=1, dim=-1)
        
    #     if self.save_router_stats and self.training:
    #         # assert self.step_ep not in self.expert_mask_per_step
    #         if self.step_ep % self.save_steps_router_stats or self.step_ep == 0:
    #             self.expert_mask_per_step[self.step_ep] = expert_mask_full.cpu().to(torch.float32).detach() #.numpy().tolist()
    #         self.step_ep += 1
    #         # with open("./expert_mask_per_step.json", 'w') as f:
    #         #     json.dump(self.expert_mask_per_step, f, indent=4)
        
    #     density_1 = torch.mean(expert_mask_full, dim=0)  # shape: [L]
    #     # For each core, get fraction of probability mass assigned to each expert
    #     # from the router across all tokens.
    #     # density 1 proxy shape: [num cores, num experts]
    #     density_1_proxy = torch.mean(router_probs, dim=0)  # shape: [L]
    #     # density l for a single core: vector of length num experts that sums to 1.
    #     # density l proxy for a single core: vector of length num experts that sums to 1.
    #     # Want both vectors to have uniform allocation (1/num experts) across all num expert elements.
    #     # The two vectors will be pushed towards uniform allocation when the dot product is minimized.
    #     loss = torch.mean(density_1_proxy * density_1) * (self.num_layers**2)
    #     return loss # add to graph so remove .item()

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
        # rank0_print(p_other_sum)
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
        # rank0_print("hidden_staes", hidden_states.shape)
        # rank0_print("weights: ", self.weights.shape)
        # #import pdb; pdb.set_trace()
        # _weight_fn is an nn.Module, so calling it via __call__ lets DeepSpeed ZeRO-3
        # attach its gather/scatter hooks for both forward AND backward automatically.
        router_logits = self._weight_fn(hidden_states)
       
        if self.training:
            # #import pdb; pdb.set_trace()
            rank0_print("rank of router_logits BEFORE noise", torch.linalg.matrix_norm(router_logits))
            # Use the device from hidden_states to avoid issues with offloaded weights
            weights_device = hidden_states.device
            random_noise = torch.rand(router_logits.shape).to(device=weights_device)
            router_logits += random_noise # add random noise to enable exploration to avoid collapse in behavior where always choose same expert
            rank0_print("rank of router_logits AFTER noise", torch.linalg.matrix_norm(router_logits))
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
            rank0_print("the variance across the layers of the non_reserve_probs are: ", non_reserved_probs.var(dim=0))
            rank0_print("the norm of non_reserved_probs", torch.linalg.matrix_norm(non_reserved_probs))
            
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
            
            rank0_print("the variance across the layers of the reserve_layer_prob are: ", reserve_layer_prob.var(dim=0))
            rank0_print("the norm of reserve_layer_prob", torch.linalg.vector_norm(reserve_layer_prob))
            router_probs = non_reserved_probs * (1.0 - reserve_layer_prob).unsqueeze(-1)
            router_probs[:, reserve_idx] = reserve_layer_prob
            router_probs = router_probs.to(router_logits.dtype)
            rank0_print("variance of router_probs across layers: ", router_probs.var(dim=0))
            rank0_print("the norm of router_probs", torch.linalg.matrix_norm(router_probs))
        else:
            router_probs = torch.softmax(router_logits, dim=-1).to(router_logits.dtype)
        
        if self.save_router_stats and self.training:
            # rank0_print("self.router_info : ", self.router_info)
            # import sys; sys.exit()
            if self.step_rp % self.save_steps_router_stats or self.step_rp == 0:
                self.router_probs_per_step[self.step_rp] = router_probs.cpu().to(torch.float32).detach() #.numpy().tolist()
            self.step_rp += 1
        # with open("/shared/nas2/sstoica2/MoL/router_probs_per_step.json", 'w') as f:
        #     json.dump(self.router_probs_per_step, f, indent=4)
        # rank0_print("router_probs.shape: ", router_probs.shape)
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
                remaining_mass.unsqueeze(-1) / probs_sum.clamp(min=1e-12),
                torch.zeros_like(probs_sum),
            )
            # NOTE: we are just adjusting the scale of probs to sum to (1 - reserve_layer_prob), 
            # so that when we add the reserve_layer_prob later, the total sum is 1.
            probs = probs * scale
            return layers, probs, reserve_layer_prob  # [batch_size, k], [batch_size, k ], [batch_size ]
        
        return layers, probs # [batch_size, k], [batch_size, k  ]

    def forward(self, hidden_states, type="clip"):
        """Produce the combine and dispatch tensors used for sending and
        receiving tokens from their highest probability expert. """
        # inputs shape: [num cores, tokens per core, d model]
        # hidden states are of size:  torch.Size([num_layers (L), batch_size (B), num_tokens (N), hidden_size (H)]). 
        L, B, N, H = hidden_states.shape
        k = self.num_layers_to_choose
        self.num_layers = L
        router_probs = self.router_probs(hidden_states)
        if type == "clip" or "siglip" in type:
            self._latest_router_probs_for_aux_clip = router_probs
        else:
            self._latest_router_probs_for_aux_dino = router_probs
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
        # rank0_print("hidden_states_chosen: ", hidden_states_chosen.shape)
        # rank0_print("last_hidden_state: ", hidden_states_chosen.shape)
        probs = probs.to(hidden_states_chosen.dtype) # convert from float32 back to bfloat16
        image_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        # rank0_print("image_interm_features: ", image_interm_features.shape)
        if k > 1:
            image_features = image_features.sum(dim=1, keepdim=True) # need to get [B, 1, N, H]

        if self.force_reserve_layer:
            # rank0_print("reserve_layer_prob.reshape(B, -1, 1, 1): ", reserve_layer_prob.reshape(B, -1, 1, 1).shape)
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
        attn_level: str = "layer", # "token", "layer", or "patch_layer" (used when text_conditioning == "attention")
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
        self.config = config
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
        # #import pdb; pdb.set_trace()
        self.text_hidden_size = self._infer_text_hidden_size(config)
        if self.text_hidden_size is None:
            raise ValueError(
                "Unable to infer text hidden size for TextConditionedRouter. "
                "Provide config.mm_text_hidden_size or config.text_config.hidden_size."
            )
        # Conditioning modules.
        if self.text_conditioning not in ("film", "attention", "attn"):
            raise ValueError(f"Unknown text_conditioning: {self.text_conditioning}")
        if self.attn_level not in ("token", "layer", "patch_layer"):
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
        if "clip" in config.encoder_type or "siglip" in config.encoder_type:
            for name in ("mm_text_hidden_size", "text_hidden_size"):
                value = getattr(config, name, None)
                if value is not None:
                    return value
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                value = getattr(text_config, "hidden_size", None)
                if value is not None:
                    return value
        else:
            value = getattr(config, "dino_mm_text_hidden_size", None)
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
        for module_name in ("text_to_film", "text_attn_mlp", "text_attn_q", "text_attn_k", "reserve_gate"):
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

    def _condition_hidden_states_attention(self, hidden_states, text_embeds, text_attention_mask):
        original_hidden_states = hidden_states
        text_summary = self._pool_text(text_embeds, text_attention_mask).to(
            device=hidden_states.device
        )
        hidden_states, text_summary = self._cast_router_projection_inputs(
            hidden_states, text_summary
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
            # for clip scores.min() is (-2.2908 while max is 3.3197
            attn = torch.softmax(scores, dim=-1).to(
                device=original_hidden_states.device,
                dtype=original_hidden_states.dtype,
            )
            return original_hidden_states * attn.unsqueeze(-1) #[T, B]

        if self.attn_level == "layer":
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
            attn = torch.softmax(scores, dim=0).to(
                device=original_hidden_states.device,
                dtype=original_hidden_states.dtype,
            )  # [L, B]
            return original_hidden_states * attn.unsqueeze(-1).unsqueeze(-1)

        raise ValueError(
            f"attn_level={self.attn_level} is not supported by hidden-state conditioning; "
            "use patch_layer routing path in forward."
        )
    
    def _patch_layer_router_probs(self, hidden_states, text_embeds, text_attention_mask):
        """Compute per-patch distributions over layers.

        Returns:
            router_probs: [B, N, L] layer probabilities for each patch token.
            router_probs_for_selection: [B, N, L] probabilities with reserved layer masked out when enabled.
            reserve_layer_prob: [B, N] reserved-layer mixing weight for each patch.
        """
        rank0_print("\n ====== hidden_states.min()", hidden_states.min(), "hidden_states.max()", hidden_states.max())
        text_summary = self._pool_text(text_embeds, text_attention_mask).to(
            device=hidden_states.device
        )
        hidden_states, text_summary = self._cast_router_projection_inputs( # torch.bfloat16
            hidden_states, text_summary
        )
        if self.text_scale is not None:
            text_summary = text_summary * self.text_scale

        if self.text_attention_type == "concat":
            text_expanded = text_summary.unsqueeze(0).unsqueeze(2)
            text_expanded = text_expanded.expand(
                hidden_states.shape[0], -1, hidden_states.shape[2], -1
            )
            concat = torch.cat([hidden_states, text_expanded], dim=-1)
            scores = self.text_attn_mlp(concat).squeeze(-1)  # [L, B, N]
        else:
            q = self.text_attn_q(text_summary)
            k = self.text_attn_k(hidden_states)  # [L, B, N, D]
            scores = (k * q.unsqueeze(0).unsqueeze(2)).sum(dim=-1) * self.text_attn_scale  # [L, B, N]
        rank0_print("\n ====== q.min()", q.min(), "q.max()", q.max())
        rank0_print("\n ====== k.min()", k.min(), "k.max()", k.max())
        rank0_print("\n ====== scores.min()", scores.min(), "scores.max()", scores.max())
        # Reorder for per-patch routing: softmax over layers for each (B, N).
        # #import pdb; pdb.set_trace()
        scores = scores.permute(1, 2, 0).to(torch.float32)  # [B, N, L] <- only part that really diff
        # scores.min() is -2.8906 and max is 2.6250 for clip; and -11.6250 and 15.4375 for dino
        if self.training:
            random_noise = torch.rand(scores.shape, device=scores.device)
            scores = scores + random_noise
        rank0_print("\n ====== scores AFTER noise min()", scores.min(), "scores.max()", scores.max())
        temperature = self.router_temperature if self.router_temperature is not None else 1.0
        if temperature != 1.0:
            scores = scores / temperature

        if self.learnable_gate and self.confidence_gate:
            raise ValueError("learnable_gate and confidence_gate are mutually exclusive.")

        use_gate = self.force_reserve_layer and (self.learnable_gate or self.confidence_gate)
        if use_gate:
            reserve_idx = self._resolve_reserve_idx(scores.shape[-1])
            masked_scores = scores.clone()
            masked_scores[:, :, reserve_idx] = -1e9
            non_reserved_probs = torch.softmax(masked_scores, dim=-1) # min is 0 and max is 0.1316 for for clip

            if self.learnable_gate:  
                if self.reserve_gate is None or self.reserve_gate.in_features != hidden_states.shape[-1]:
                    self.reserve_gate = nn.Linear(hidden_states.shape[-1], 1).to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                reserve_hidden = hidden_states[self.reserve_layer]  # [B, N, H]
                # #import pdb; pdb.set_trace()
                try:
                    gate_logits = self.reserve_gate(reserve_hidden).squeeze(-1) # min is -2.4062 and max is 1.4141
                except:
                    gate_logits = self.reserve_gate.to(torch.float32)(reserve_hidden).squeeze(-1)
                rank0_print("\n ====== gate_logits.min()", gate_logits.min(), "gate_logits.max()", gate_logits.max())
                reserve_layer_prob = torch.sigmoid(gate_logits)  # [B, N] # min is 0 and max is 0.0935
            else:
                conf = non_reserved_probs.max(dim=-1).values
                reserve_layer_prob = torch.sigmoid(
                    self.confidence_gate_slope * (self.confidence_gate_threshold - conf)
                )
            # #import pdb; pdb.set_trace()
            router_probs = non_reserved_probs * (1.0 - reserve_layer_prob).unsqueeze(-1)
            router_probs[:, :, reserve_idx] = reserve_layer_prob
            router_probs = router_probs.to(scores.dtype)
        else:
            router_probs = torch.softmax(scores, dim=-1).to(scores.dtype)  # [B, N, L] 
        rank0_print("\n ====== router_probs.min()", router_probs.min(), "router_probs.max()", router_probs.max())
        if self.save_router_stats and self.training:
            router_probs_summary = router_probs.mean(dim=1)  # [B, L]
            if self.step_rp % self.save_steps_router_stats or self.step_rp == 0:
                self.router_probs_per_step[self.step_rp] = (
                    router_probs_summary.cpu().to(torch.float32).detach()
                )
            self.step_rp += 1

        if self.cap_softmax:
            B, N, L = router_probs.shape
            router_probs = self.cap_layer_prob(
                router_probs.reshape(B * N, L),
                self.cap_softmax_value,
                self.reserve_layer,
            ).reshape(B, N, L)

        if self.force_reserve_layer:
            reserve_idx = self._resolve_reserve_idx(router_probs.shape[-1])
            router_probs_for_selection = router_probs.clone() # min is 0.0015 and max is 0.8047 for clip
            router_probs_for_selection[:, :, reserve_idx] = -1.0
            reserve_layer_prob = router_probs[:, :, reserve_idx]
            rank0_print("\n ======= router_probs_for_selection.min()", router_probs_for_selection.min(), "router_probs_for_selection.max()", router_probs_for_selection.max())
            rank0_print("\n ======= reserve_layer_prob.min()", reserve_layer_prob.min(), "reserve_layer_prob.max()", reserve_layer_prob.max())
            rank0_print("\n ================================================")
            return router_probs, router_probs_for_selection, reserve_layer_prob
        rank0_print("\n ================================================")
        return router_probs, router_probs, None # ths is float32
    
    def _get_patch_layer_and_scaling_probs(
        self,
        router_probs: torch.Tensor,
        router_probs_for_selection: torch.Tensor,
        reserve_layer_prob: Optional[torch.Tensor] = None,
    ):
        #import pdb; pdb.set_trace()
        available_layers = router_probs.shape[-1] - (1 if self.force_reserve_layer else 0)
        k = min(self.num_layers_to_choose, available_layers)
        rank0_print("\n k:", k)
        rank0_print("\n router_probs_for_selection min ", router_probs_for_selection.min(), "max", router_probs_for_selection.max())
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
            rank0_print("\n getting patch_layer_and_scaling_probs with forced reserve layer...")
            rank0_print("\n remaining_mass min ", remaining_mass.min(), "max", remaining_mass.max())
            rank0_print("\n probs_sum min ", probs_sum.min(), "max", probs_sum.max())
            scale = torch.where(
                probs_sum > 0,
                remaining_mass.unsqueeze(-1) / probs_sum.clamp(min=1e-12),
                torch.zeros_like(probs_sum),
            )
            rank0_print("\n scale min ", scale.min(), "max ", scale.max())
            probs = probs * scale
            rank0_print("\n probs min ", probs.min(), "max ", probs.max())
            rank0_print("\n =================================================")

        return layers, probs, reserve_layer_prob

    def _condition_hidden_states(self, hidden_states, text_embeds, text_attention_mask):
        if self.text_conditioning == "film":
            return self._condition_hidden_states_film(hidden_states, text_embeds, text_attention_mask)
        return self._condition_hidden_states_attention(hidden_states, text_embeds, text_attention_mask)

    def router_probs(self, hidden_states, text_embeds=None, text_attention_mask=None):
        assert text_embeds is not None, "\n we are using text conditioning so text_embeds should not be none"
        if text_embeds is None:
            return super().router_probs(hidden_states)
        if (
                self.text_conditioning in ("attention", "attn")
                and self.attn_level == "patch_layer"
            ):  # patch-level, layer-wise attention conditioning with differentiable per-patch layer mixing
            rank0_print("\n computing patchwise attention....")
            patch_router_probs, _, _ = self._patch_layer_router_probs(
                hidden_states, text_embeds, text_attention_mask
            )
            # Keep API compatible with [B, L] expected by auxiliary loss code.
            return patch_router_probs.mean(dim=1)
        rank0_print("\n text conditioned the visual hidden states....")
        conditioned_hidden_states = self._condition_hidden_states(
            hidden_states, text_embeds, text_attention_mask
        )
        # #import pdb; pdb.set_trace()
        return super().router_probs(conditioned_hidden_states)

    def forward(self, hidden_states, text_embeds=None, text_attention_mask=None, type="clip"):
        """Route vision layers conditioned on text embeddings.

        Args:
            hidden_states: [L, B, N, H] vision hidden states.
            text_embeds: [B, T, H_text] or [B, H_text] text embeddings.
            text_attention_mask: [B, T] attention mask for pooling.
        """
        L, B, N, H = hidden_states.shape
        # #import pdb; pdb.set_trace()
        if (
            text_embeds is not None
            and self.text_conditioning in ("attention", "attn")
            and self.attn_level == "patch_layer"
        ):  # patch-level attention conditioning with per-patch top-k layer routing
            #import pdb; pdb.set_trace()
            rank0_print("\n calculating forward using patch_layer branch...")
            self.num_layers = L
            router_probs, router_probs_for_selection, reserve_layer_prob = self._patch_layer_router_probs(
                hidden_states, text_embeds, text_attention_mask
            )
            # Cache the per-patch router probabilities for auxiliary loss.
            if type == "clip" or type == "siglip":
                self._latest_router_probs_for_aux_clip = {
                    "kind": "patch_layer",
                    "router_probs": router_probs,
                    "router_probs_for_selection": router_probs_for_selection,
                    "reserve_layer_prob": reserve_layer_prob,
                }
            else:
                self._latest_router_probs_for_aux_dino = {
                    "kind": "patch_layer",
                    "router_probs": router_probs,
                    "router_probs_for_selection": router_probs_for_selection,
                    "reserve_layer_prob": reserve_layer_prob,
                }

            hidden_states = hidden_states.permute(1, 2, 0, 3)  # [B, N, L, H]
            # layers is torch.Size([32, 257, 1]), probs is torch.Size([32, 257, 1]), reserve_layer_prob is torch.Size([32, 257])
            layers, probs, reserve_layer_prob = self._get_patch_layer_and_scaling_probs(
                router_probs,
                router_probs_for_selection,
                reserve_layer_prob,
            )
            # #import pdb; pdb.set_trace()
            # rank0_print("\n layers: ", layers.transpose(0,1))
            # self._cache_latest_routing_decision(
            #     "patch_level",
            #     layers,
            #     probs,
            #     reserve_layer_prob=reserve_layer_prob,
            # )
            gather_idx = layers.unsqueeze(-1).expand(-1, -1, -1, H) # [B, N, 1, H]
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

        k = self.num_layers_to_choose
        rank0_print("k : ", k)
        self.num_layers = L
        router_probs = self.router_probs(hidden_states, text_embeds, text_attention_mask)
        if type == "clip" or type == "siglip":
            self._latest_router_probs_for_aux_clip = router_probs
        else:
            self._latest_router_probs_for_aux_dino = router_probs
        
        if self.force_reserve_layer:
            layers, probs, reserve_layer_prob = self.get_layer_and_scaling_probs(router_probs)
        else:
            layers, probs = self.get_layer_and_scaling_probs(router_probs)
        if type == "clip" or type == "siglip":
            self.layers_clip = layers
        if type == "dino":
            self.layers_dino = layers
        rank0_print("layers:", layers.transpose(0,1))
        # #import pdb; pdb.set_trace()
        hidden_states = hidden_states.transpose(0, 1)
        layers = layers.reshape(B, -1, 1, 1).expand(B, k, N, H) # [B, k, N, H]        
        hidden_states_chosen = torch.gather(hidden_states, 1, layers)  # [B, k, N, H]     
        if self.force_reserve_layer:
            reserve_hidden_state = hidden_states[:, self.reserve_layer, :, :].unsqueeze(1) # [B, 1, N, H]
        # rank0_print("hidden_states_chosen: ", hidden_states_chosen.shape)
        # rank0_print("last_hidden_state: ", hidden_states_chosen.shape)
        probs = probs.to(hidden_states_chosen.dtype) # convert from float32 back to bfloat16
        image_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        # rank0_print("image_interm_features: ", image_interm_features.shape)
        if k > 1:
            image_features = image_features.sum(dim=1, keepdim=True) # need to get [B, 1, N, H]

        if self.force_reserve_layer:
            # rank0_print("reserve_layer_prob.reshape(B, -1, 1, 1): ", reserve_layer_prob.reshape(B, -1, 1, 1).shape)
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
        if "clip" in config.encoder_type or "siglip" in config.encoder_type:
            for name in ("mm_text_hidden_size", "text_hidden_size"):
                value = getattr(config, name, None)
                if value is not None:
                    return value
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                value = getattr(text_config, "hidden_size", None)
                if value is not None:
                    return value
        else:
            value = getattr(config, "dino_mm_text_hidden_size", None)
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

    def forward(self, hidden_states, text_embeds=None, text_attention_mask=None, type="clip"):
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
        if type == "clip" or type == "siglip":
            self._latest_router_probs_for_aux_clip = router_probs
        else:
            self._latest_router_probs_for_aux_dino = router_probs
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
        image_features = self._select_vision_features(image_features)
        assert image_features.ndim == 3
        return image_features
        
