import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                rank0_print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    # import pdb; pdb.set_trace()
    to_return = {k: t for k, t in named_params if any(k == key or k.startswith(key + ".") or k == f"model.{key}" or k.startswith(f"model.{key}.") for key in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, encoder_type, fusion_type_when_preparing_inputs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_type = encoder_type
        self.fusion_type = fusion_type_when_preparing_inputs


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        # rank0_print(f"[DeepSpeed check] self.optimizer type: {type(self.optimizer)}")
        # rank0_print(f"[DeepSpeed check] optimizer: {self.optimizer.__dict__}")
        # import sys; sys.exit(0)
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name and "scaling" not in name]
            scalar_parameters = [n for n, p in opt_model.named_parameters() if "scaling" in n and p.requires_grad]
            
            optimizer_grouped_parameters = []
            if self.args.mm_projector_lr is not None or self.args.dino_mm_projector_lr is not None:
                # set dino_mm_projector_lr and mm_projector_lr equal if one is defined and not the other
                if self.args.dino_mm_projector_lr is None:
                    self.args.dino_mm_projector_lr = self.args.mm_projector_lr
                if self.args.mm_projector_lr is None:
                    self.args.mm_projector_lr = self.args.dino_mm_projector_lr

                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    # non-projector, non-scaling, non-router params with decay
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and n not in scalar_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },

                    # scaling params: no weight decay (optionally add a smaller lr here)
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in scalar_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr * 0.1,  # optional
                    },

                    # non-projector, non-scaling, non-router params without decay
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and n not in scalar_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },

                    # projector params with decay (mm_projector_lr)
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad and "dino" not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad and "dino" in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.dino_mm_projector_lr,
                    },

                    # projector params without decay (mm_projector_lr)
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad and "dino" not in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad and "dino" in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.dino_mm_projector_lr,
                    },
                ]

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            # import pdb; pdb.set_trace()
            rank0_print("TRAINABLE OPTIMIZER")
            # import pdb; pdb.set_trace()
            id2name = {id(p): n for n, p in opt_model.named_parameters()}
            for gi, g in enumerate(optimizer_grouped_parameters):
                if len(g["params"]) == 0:
                    continue
                lr = g.get("lr", self.args.learning_rate)
                wd = g.get("weight_decay", 0.0)
                try:
                    rank0_print(f"\n=== group {gi}: lr={lr:.3e} wd={wd} ===")
                except:
                    import pdb; pdb.set_trace()
                for p in g["params"]:
                    name = id2name.get(id(p), "<unknown>")
                    rank0_print(" ", name)
            # import sys; sys.exit(0)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        # we need to manually add the deepseed logic for saving else we run into the "no valid checkpoint"
        # to provide background: because LLaVA overised the _save_checkpoint function and removes the 
        # deepseed save call, "global_step" is not being saved in the checkpoints, resulting in deepspeed
        # becing unable to restart.
        if self.is_deepspeed_enabled:
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            # have deepspeed save what it needs to
            self.deepspeed.save_checkpoint(output_dir)
        
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self._save_optimizer_and_scheduler(output_dir)  # NOTE: save optimizer and scheduler states with deepspeed

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if len(weight_to_save) > 0 and (self.args.local_rank == 0 or self.args.local_rank == -1):
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            
            keys_to_match = ['dino_mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            
            # import pdb; pdb.set_trace()
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if (self.args.local_rank == 0 or self.args.local_rank == -1) and len(weight_to_save) > 0:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'dino_mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict) # problem here

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: torch.Tensor | None = None):
        # print("============================= compute_loss =============================")
        inputs["encoder_type"] = self.encoder_type
        inputs["fusion_type"] = self.fusion_type
        
        # input is of type dict dict_keys(['input_ids', 'labels', 'attention_mask', 'clip_texts', 'images', 'dino_images', 'encoder_type', 'interleave']) 
        # import pdb; pdb.set_trace()
        images = inputs['images'] 
        dino_images = inputs['dino_images'] if 'dino_images' in inputs else None
        if dino_images is None and "dino" in self.encoder_type: # I want to be very clear what images are being passed
            dino_images = images
            inputs['dino_images'] = dino_images
            images = None
            inputs['images'] = images
        import pdb; pdb.set_trace()
        outputs = model(**inputs)
        loss = outputs.loss
        assert torch.isnan(loss) == False, "Loss is NaN!"
        return (loss, outputs) if return_outputs else loss

# =========================================================================== #
class RouterLLaVATrainer(Trainer):
    def __init__(self, encoder_type, fusion_type_when_preparing_inputs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_type = encoder_type
        self.fusion_type = fusion_type_when_preparing_inputs
    

    aux_losses = []
    last_aux_loss: float = 0.0

    def get_aux_losses_sep(self):
        import json
        with open("/shared/nas2/sstoica2/LLaVA/aux_loss.json", 'w') as f:
            json.dump(self.aux_losses, f, indent=4)
        return self.aux_losses
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # import pdb; pdb.set_trace()
        if (self.train_dataset is None or not has_length(self.train_dataset)) and (self.train_dataset_dino is None or not has_length(self.train_dataset_dino)) :
            return None
        
        # # make dino training dataset the dataset if the main training dataset is not provided
        # if self.train_dataset is None and self.train_dataset_dino is not None:
        #     self.train_dataset = self.train_dataset_dino

        # can I force the batch size for both clip and dino to be the same? 
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # import pdb; pdb.set_trace()
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name and "scaling" not in name]
            scalar_parameters = [n for n, p in opt_model.named_parameters() if "scaling" in n and p.requires_grad]
            router_parameters = [n for n, p in opt_model.named_parameters() if "router" in n and p.requires_grad]

            optimizer_grouped_parameters = []
            if self.args.mm_projector_lr is not None or self.args.dino_mm_projector_lr is not None:
                # set dino_mm_projector_lr aned mm_projector_lr equal if one is defined and not the other
                if self.args.dino_mm_projector_lr is None:
                    self.args.dino_mm_projector_lr = self.args.mm_projector_lr
                if self.args.mm_projector_lr is None:
                    self.args.mm_projector_lr = self.args.dino_mm_projector_lr
                
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    # non-projector, non-scaling, non-router params with decay
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and n not in scalar_parameters and n not in router_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },

                    # scaling params: no weight decay (optionally add a smaller lr here)
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in scalar_parameters and n not in router_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr * 0.1,  # optional
                    },

                    # non-projector, non-scaling, non-router params without decay
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and n not in scalar_parameters and n not in router_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },

                    # projector params with decay (mm_projector_lr)
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad and "dino" not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad and "dino" in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.dino_mm_projector_lr,
                    },

                    # projector params without decay (mm_projector_lr)
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad and "dino" not in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad and "dino" in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.dino_mm_projector_lr,
                    },
                ]

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in router_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in router_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            if self.args.router_lr is not None:
                optimizer_grouped_parameters.append(
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in router_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.router_lr,
                    },)
            else:
                optimizer_grouped_parameters.append(
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in router_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },)
            rank0_print("TRAINABLE OPTIMIZER")
            # import pdb; pdb.set_trace()
            id2name = {id(p): n for n, p in opt_model.named_parameters()}
            for gi, g in enumerate(optimizer_grouped_parameters):
                if len(g["params"]) == 0:
                    continue
                lr = g.get("lr", self.args.learning_rate)
                wd = g.get("weight_decay", 0.0)
                try:
                    rank0_print(f"\n=== group {gi}: lr={lr:.3e} wd={wd} ===")
                except:
                    import pdb; pdb.set_trace()
                for p in g["params"]:
                    name = id2name.get(id(p), "<unknown>")
                    rank0_print(" ", name)
            # import sys; sys.exit(0)
            # all_group_params = set()
            # for group in optimizer_grouped_parameters:
            #     for p in group["params"]:
            #         all_group_params.add(p)

            # all_trainable = {p for _, p in opt_model.named_parameters() if p.requires_grad}

            # print("Missing params:", all_trainable - all_group_params)
            # print("Extra params:", all_group_params - all_trainable)
            # import pdb; pdb.set_trace()
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        # we need to manually add the deepseed logic for saving else we run into the "no valid checkpoint"
        # to provide background: because LLaVA overised the _save_checkpoint function and removes the 
        # deepseed save call, "global_step" is not being saved in the checkpoints, resulting in deepspeed
        # becing unable to restart.
        if self.is_deepspeed_enabled:
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            # have deepspeed save what it needs to
            self.deepspeed.save_checkpoint(output_dir)
        
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self._save_optimizer_and_scheduler(output_dir)  # NOTE: save optimizer and scheduler states with deepspeed

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler', 'router_clip', 'clip_dino_fusion']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if len(weight_to_save) > 0 and (self.args.local_rank == 0 or self.args.local_rank == -1):
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            
            keys_to_match = ['dino_mm_projector', 'router_dino']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            
            # import pdb; pdb.set_trace()
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if (self.args.local_rank == 0 or self.args.local_rank == -1) and len(weight_to_save) > 0:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'dino_mm_projector.bin'))
        else:
            super(RouterLLaVATrainer, self)._save_checkpoint(model, trial, metrics)
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            # Ensure generation_config is valid before save_pretrained
            gen_cfg = getattr(self.model, "generation_config", None)
            if gen_cfg is not None:
                do_sample = getattr(gen_cfg, "do_sample", None)
                temperature = getattr(gen_cfg, "temperature", None)
                top_p = getattr(gen_cfg, "top_p", None)
                if do_sample is False:
                    if temperature is None or temperature != 1.0:
                        gen_cfg.temperature = 1.0
                    if top_p is None or top_p != 1.0:
                        gen_cfg.top_p = 1.0
            super(RouterLLaVATrainer, self)._save(output_dir, state_dict) # problem here

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: torch.Tensor | None = None):
        # print("============================= compute_loss =============================")
        inputs["encoder_type"] = self.encoder_type
        inputs["fusion_type"] = self.fusion_type
        # input is of type dict dict_keys(['input_ids', 'labels', 'attention_mask', 'clip_texts', 'images', 'dino_images', 'encoder_type', 'interleave']) 
        # import pdb; pdb.set_trace()
        images = inputs['images'] 
        dino_images = inputs['dino_images'] if 'dino_images' in inputs else None
        # import pdb; pdb.set_trace()
        if dino_images is None and "dino" in self.encoder_type: # I want to be very clear what images are being passed
            dino_images = images
            inputs['dino_images'] = dino_images
            images = None
            inputs['images'] = images

        outputs = model(**inputs)
        loss = outputs.loss
        rank0_print(f"\n ================ main loss: {loss.item():.4f}")
        # print("==================== progressing to get the aux_loss ====================")
        # import pdb; pdb.set_trace()
        if images is not None and dino_images is not None:
            load_balance_loss = model.get_aux_loss(images, dino_images)
        else:
            if dino_images is not None and images is None:
                images = dino_images
            load_balance_loss = model.get_aux_loss(images)
        rank0_print(f"------------------------ END ONE STEP ------------------------")
        self.aux_losses.append(load_balance_loss)
        self.last_aux_loss = float(load_balance_loss.detach().cpu().item())
        loss += model.load_balance_alpha * load_balance_loss
        assert torch.isnan(loss) == False, "Loss is NaN!"
        # if getattr(model, "load_balance_alpha", 0.0) != 0.0:
        #     images = inputs.get('images')
        #     # print("==================== progressing to get the aux_loss ====================")
        #     load_balance_loss = model.get_aux_loss(images)
        #     self.aux_losses.append(load_balance_loss)
        #     loss += model.load_balance_alpha * load_balance_loss
        return (loss, outputs) if return_outputs else loss

