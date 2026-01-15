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
        self.vision_feature_select_strategy = "default" # 
        self.weights = nn.Parameter(weights) #[num_tokens, hidden_size]
        self.load_balancing_alpha = load_balancing_alpha
        # self.bias = nn.Parameter(torch.zeros(24)) # typically not used
        # usually MoE have a capacity parameter which restricts how many tokens a expert gets, but 
        # do not do that here, so I removed it

    def load_balance_loss(self, router_probs, expert_mask):
        """Calculate load−balancing loss to ensure diverse expert routing. Taken from swift transformer pseudo code
        modified to hard code num_experts because we know it is 24, but can find it dynamically too easily.
        Also, removed tensor_cores because we do not need to restrict input number of tokens to any expert 
        
        Args:
            router_probs: [batch_size, num_layers] - probability assigned to each expert per image
            expert_mask: [batch_size, num_layers] - one-hot vector indicating which layer was actually selected
        """
        # router probs is the probability assigned for each expert per image.
        # expert mask contains the expert with the highest router probability in one−hot format.
        # router_probs and expert_mask = [batch_size, num_layers]
        assert self.num_layers is not None
        # print("expert_mask: ", expert_mask.dtype, expert_mask.shape)
        # print("router_probs: ", router_probs.dtype, router_probs.shape)
        density_1 = torch.mean(expert_mask, dim=0)  # shape: [L]
        # For each core, get fraction of probability mass assigned to each expert
        # from the router across all tokens.
        # density 1 proxy shape: [num cores, num experts]
        density_1_proxy = torch.mean(router_probs, dim=0)  # shape: [L]
        # density l for a single core: vector of length num experts that sums to 1.
        # density l proxy for a single core: vector of length num experts that sums to 1.
        # Want both vectors to have uniform allocation (1/num experts) across all num expert elements.
        # The two vectors will be pushed towards uniform allocation when the dot product is minimized.
        loss = torch.mean(density_1_proxy * density_1) * (self.num_layers**2)
        return loss

    def router_probs(self, hidden_states):
        # hidden states are of size:  torch.Size([num_layers, batch_size, num_tokens, hidden_size]).
        # print("hidden_states: ", hidden_states.shape)
        # print("self.weights: ", self.weights.shape)
        router_logits = torch.einsum("lbnh, nh ->bl", hidden_states, self.weights) # torch.Size([batch_size, num_layers])

        if self.training:
            random_noise = torch.rand(router_logits.shape).to(device=self.weights.device)
            router_logits += random_noise # add random noise to enable exploration to avoid collapse in behavior where always choose same expert
       
        # Convert input to softmax operation from bfloat16 to float32 for stability. <- do I need to do this?
        #router logits = mtf.to float32(router logits)

        # TODO: reshape to get num_layers in last dim
        router_probs = torch.softmax(router_logits, dim=-1)
        # print("router_probs.shape: ", router_probs.shape)
        return router_probs # torch.Size([32, 24])

    def get_layer_and_scaling_probs_and_expert_mask(self, router_probs):
        # the last dim in router probs is num_layers: 24
        # router_probs = [batch_size, num_layers]
        sorted_probs, sorted_idxs = torch.sort(router_probs[:, :-1], dim=-1, descending=True) # remove the last layer from consideration
        probs = sorted_probs[:, :self.num_layers_to_choose]
        layers = sorted_idxs[:, :self.num_layers_to_choose]
        last_layer_prob = router_probs[:, -1]
        # build expert mask (top-k safe): [B, k, L] -> [B, L]
        expert_mask = F.one_hot(layers, num_classes=self.num_layers).float().sum(dim=1)
        # normalize so each row sums to 1 (important for top-k > 1) -> routing mass needs to be distributed to sum up to 1
        expert_mask = expert_mask / expert_mask.sum(dim=-1, keepdim=True)
        # print("layers, probs, last_layer_prob, expert_mask: ", [x.shape for x in [layers, probs, last_layer_prob, expert_mask]])
        return layers, probs, last_layer_prob, expert_mask # [batch_size, num_layers_to_choose] x2, [batch_size, ], [batch_size, num_layers]

    def forward(self, hidden_states):
        """Produce the combine and dispatch tensors used for sending and
        receiving tokens from their highest probability expert. """
        # Core layout is split across num cores for all tensors and operations.
        # inputs shape: [num cores, tokens per core, d model]
        # hidden states are of size:  torch.Size([num_layers, batch_size, num_tokens, hidden_size]). 16 is batch size --> we need to figure out how to get the 577
        # print("config:",  self.config)
        L, B, N, H = hidden_states.shape
        k = self.num_layers_to_choose
        self.num_layers = L
        router_probs = self.router_probs(hidden_states)
        layers, probs, last_layer_prob, _ = self.get_layer_and_scaling_probs_and_expert_mask(router_probs)

        # use torch.gather to gather the indexes in layer to get image_features of shape [B, L, N, H]
        hidden_states = hidden_states.transpose(0, 1)
        layers = layers.reshape(B, -1, 1, 1).expand(B, k, N, H)      
        hidden_states_chosen = torch.gather(hidden_states, 1, layers)
        last_hidden_state = hidden_states[:, -1, :, :].unsqueeze(1)

        image_interm_features = probs.reshape(B, -1, 1, 1) * hidden_states_chosen
        if k > 1:
            image_interm_features = image_interm_features.sum(dim=1, keepdim=True) # need to get [B, 1, N, H]

        image_features = image_interm_features + last_layer_prob.reshape(B, -1, 1, 1) * last_hidden_state

        # print("image feature after aggregation: ", image_features.shape) # [B, 1, N, H]
        image_feature = image_features.squeeze() # [B, N, H]
        if self.vision_feature_select_strategy == "patch": 
            image_feature = image_features[:, 1:, :]
        if self.vision_feature_select_strategy == "cls": 
            image_feature = image_features[:, 0, :]
        assert image_feature.ndim == 3
        return image_feature
