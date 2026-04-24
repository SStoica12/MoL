import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.router_logging import (
    clear_router_selection_metadata,
    get_router_selection_metadata,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_files = line.get("images")
        if image_files is None:
            image_file = line.get("image")
            image_files = [image_file] if image_file is not None and str(image_file).strip() != "" else []
        else:
            image_files = [image_file for image_file in image_files if image_file is not None and str(image_file).strip() != ""]
        has_image = len(image_files) > 0
        qs = line["text"]
        if has_image:
            replace_token = DEFAULT_IMAGE_TOKEN
            if self.model_config.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

            if IMAGE_PLACEHOLDER in qs:
                if qs.count(IMAGE_PLACEHOLDER) != len(image_files):
                    raise ValueError(
                        f"Question {line.get('question_id')} has {len(image_files)} images but "
                        f"{qs.count(IMAGE_PLACEHOLDER)} image placeholders."
                    )
                qs = qs.replace(IMAGE_PLACEHOLDER, replace_token)
            elif DEFAULT_IMAGE_TOKEN in qs:
                if qs.count(DEFAULT_IMAGE_TOKEN) != len(image_files):
                    raise ValueError(
                        f"Question {line.get('question_id')} has {len(image_files)} images but "
                        f"{qs.count(DEFAULT_IMAGE_TOKEN)} <image> tokens."
                    )
                if self.model_config.mm_use_im_start_end:
                    qs = qs.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            else:
                qs = "\n".join([replace_token] * len(image_files)) + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if has_image:
            resolved_paths = [
                image_file if os.path.isabs(image_file) else os.path.join(self.image_folder, image_file)
                for image_file in image_files
            ]
            images = []
            for image_path in resolved_paths:
                try:
                    with Image.open(image_path) as image:
                        images.append(image.convert('RGB'))
                except Exception as exc:
                    raise OSError(
                        f"Failed to load image '{image_path}' for question_id={line.get('question_id')}: {exc}"
                    ) from exc
            image_tensor = process_images(images, self.image_processor, self.model_config)
            image_sizes = [image.size for image in images]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        else:
            image_tensor = None
            image_sizes = None
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids[0]

        return input_ids, image_tensor, image_sizes, prompt

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    assert len(batch) == 1, "batch_size must be 1"
    input_ids, image_tensor, image_sizes, prompt = batch[0]
    input_ids = input_ids.unsqueeze(0)
    if image_tensor is not None:
        if hasattr(image_tensor, "ndim") and image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
    else:
        image_sizes = None
    return input_ids, image_tensor, image_sizes, prompt


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    conv_mode,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode=conv_mode,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        conv_mode=args.conv_mode,
    )

    for (input_ids, image_tensor, image_sizes, prompt), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            clear_router_selection_metadata(model)
            generate_kwargs = dict(
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            if image_tensor is not None:
                if isinstance(image_tensor, list):
                    generate_kwargs["images"] = [
                        image.to(dtype=torch.float16, device='cuda', non_blocking=True)
                        for image in image_tensor
                    ]
                else:
                    generate_kwargs["images"] = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
                generate_kwargs["image_sizes"] = image_sizes
                generate_kwargs["clip_texts"] = [prompt]
            output_ids = model.generate(
                input_ids,
                **generate_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        metadata = get_router_selection_metadata(model)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": metadata}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
