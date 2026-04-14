import torch
# from transformers import SiglipVisionConfig, CLIPVisionConfig
import glob
import os
import numpy as np
from safetensors import safe_open

vision_tower = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l'
            ).visual_model
import pdb; pdb.set_trace()

# dists = np.load("/shared/nas2/sstoica2/MoL/playground/router_distribution/router_probs_gqa_router_dino_distributin.npy") # (4192, 24)
# import pdb; pdb.set_trace()

# siglip_224 = SiglipVisionConfig.from_pretrained("google/siglip2-so400m-patch14-224")
# siglip_384 = SiglipVisionConfig.from_pretrained("google/siglip2-so400m-patch14-384")
# clip = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336")
# import pdb; pdb.set_trace()

# from huggingface_hub import snapshot_download

# local_dir = snapshot_download(
#     repo_id="BoyaWu10/Bunny-v1_1-data",
#     repo_type="dataset",          # important: this is a dataset
#     local_dir="/shared/nas2/sstoica2/Bunny/training_data/bunny_v1_1_data",# where you want it stored
#     local_dir_use_symlinks=False  # ensures real files (not refs to cache)
# )

# print(local_dir)



# vision_tower = torch.hub.load(
#                 'facebookresearch/dinov2',
#                 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l'
#             ).visual_model
# import pdb; pdb.set_trace()
# print(vision_tower.config)

# state_dict = torch.load(
#     "/shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-pretrain_2e-4_text-cond-attn-token-router_pretrain_dino/dino_mm_projector.bin",
#     map_location="cpu"
# )
# import pdb; pdb.set_trace()
# root_dir = "/shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_dino"
# # import pdb; pdb.set_trace()
# files = sorted(glob.glob(os.path.join(root_dir, "model-*-of-00006.safetensors")))

# tensors = {}
# for i, file in enumerate(files):
#     if i == len(files) - 1:
#         try:
#             with safe_open(file, framework="pt", device="cpu") as f:
#                 print(f"Metadata: {f.metadata()}") # Optional: print file metadata
#                 for key in f.keys():
#                     tensors[key] = f.get_tensor(key)
#                     print(f"Parameter Name: {key}")
#                     print(f"Value:\n{tensors[key]}\n")

#         except FileNotFoundError:
#             print(f"Error: The file '{file}' was not found.")
#         except Exception as e:
#             print(f"An error occurred: {e}")
# import pdb; pdb.set_trace()
# print(vision_tower)

# # import pdb; pdb.set_trace()
# # processor = AutoImageProcessor.from_pretrained('facebookresearch/dinov2')
# vision_tower_name = "openai/clip-vit-large-patch14-336"
# print("=============================================")
# print(CLIPVisionConfig.from_pretrained(vision_tower_name))