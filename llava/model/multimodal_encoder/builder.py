import os
from .clip_encoder import CLIPVisionTextTower, CLIPVisionTower, CLIPVisionTowerS2
from .dino_encoder import DINOVisionTower, DinoVisionTextTower

# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
#     is_absolute_path_exists = os.path.exists(vision_tower)
#     use_s2 = getattr(vision_tower_cfg, 's2', False)
#     use_text_enc = getattr(vision_tower_cfg, "mm_use_text_enc", getattr(vision_tower_cfg, "use_text_enc", False))
#     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
#         if use_text_enc:
#             return CLIPVisionTextTower(vision_tower, args=vision_tower_cfg, **kwargs)
#         elif use_s2:
#             return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
#         else:
#             return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

def build_vision_tower(vision_tower_cfg, load_model = "clip", **kwargs):
    rank0_print("vision_tower_cfg.use_text_enc", vision_tower_cfg.mm_use_text_enc)
    vision_tower_clip = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    rank0_print("Vision tower clip encoder type: ", vision_tower_clip)
    if load_model == "dino" and "clip" in vision_tower_cfg.encoder_type: # this means we have clip dino
        idx = vision_tower_cfg.encoder_type.index("dino")
        vision_tower_dino = vision_tower_cfg.encoder_type[idx:]
        rank0_print("Loading Dino Tower with clip dino encoder type: ", vision_tower_dino)
    else:
        vision_tower_dino = getattr(vision_tower_cfg, 'encoder_type')
        rank0_print("Loading Dino Tower with dino encoder type: ", vision_tower_dino)
    
    # import pdb; pdb.set_trace()
    use_text_enc = getattr(vision_tower_cfg, "mm_use_text_enc", getattr(vision_tower_cfg, "use_text_enc", False))
    rank0_print("use_text_enc : ", use_text_enc)
    # import pdb; pdb.set_trace()
    if load_model == "siglip":
        from .siglip_encoder import SiglipVisionTower, SiglipVisionTextTower
        if not use_text_enc:
            rank0_print("\n Loading SiglipVisionTower")
            return SiglipVisionTower(vision_tower_clip, args=vision_tower_cfg, **kwargs)
        else:
            rank0_print("\n Loaded SiglipVisionTextTower")
            return SiglipVisionTextTower(vision_tower_clip, args=vision_tower_cfg, **kwargs)
    elif vision_tower_dino is not None and load_model != "clip":
        if not use_text_enc:
            rank0_print("\n Loading DinoVisionTower")
            return DINOVisionTower(vision_tower_dino, args=vision_tower_cfg, **kwargs)
        else:
            rank0_print("\n Loaded DinoVisionTextTower")
            return DinoVisionTextTower(vision_tower_dino, vision_tower_clip, args=vision_tower_cfg, **kwargs)

    elif vision_tower_clip.startswith("openai") or vision_tower_clip.startswith("laion") or "ShareGPT4V" in vision_tower_clip:
        #return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        use_s2 = getattr(vision_tower_cfg, 's2', False)
        if use_text_enc:
            rank0_print("\n Loaded CLIPVisionTextTower")
            return CLIPVisionTextTower(vision_tower_clip, args=vision_tower_cfg, **kwargs)
        elif use_s2:
            rank0_print("\n Loaded CLIPVisionTowerS2")
            return CLIPVisionTowerS2(vision_tower_clip, args=vision_tower_cfg, **kwargs)
        else:
            rank0_print("\n Loaded CLIPVisionTower")
            CLIPVisionTower(vision_tower_clip, args=vision_tower_cfg, **kwargs)        
    raise ValueError(f"Unkown clip vision tower {vision_tower_clip}" if "clip" in load_model else f"Unkown dino vision tower {vision_tower_dino}")
