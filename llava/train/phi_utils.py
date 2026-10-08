"""Phi's Bunny conversation format, with assistant-only training labels.

The no-BOS round accounting follows Bunny's Phi data preprocessing:
https://github.com/BAAI-DCAI/Bunny/blob/main/bunny/util/data_utils.py
"""
from llava.constants import IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


def preprocess_phi(sources, tokenizer):
    input_ids, labels, conversations = [], [], []
    for source in sources:
        conv = conv_templates["bunny"].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        if source and source[0]["from"] != "human":
            source = source[1:]
        for index, message in enumerate(source):
            role = roles[message["from"]]
            if role != conv.roles[index % 2]:
                raise ValueError("Phi conversations must alternate human and assistant messages.")
            conv.append_message(role, message["value"])
        conversation = conv.get_prompt()
        ids = tokenizer_image_token(conversation, tokenizer, return_tensors="pt")
        target = ids.clone()
        separator = conv.sep + conv.roles[1] + ": "
        cursor = 0
        for turn in conversation.split(conv.sep2):
            if not turn:
                continue
            parts = turn.split(separator)
            if len(parts) != 2:
                raise ValueError("Phi instruction data needs a completed assistant reply per round.")
            instruction = parts[0] + separator
            # Phi's tokenizer has no automatic BOS. Its trailing prompt space
            # joins the first response token; the EOS marker is one token.
            prompt_length = len(tokenizer_image_token(instruction, tokenizer)) - 1
            round_length = len(tokenizer_image_token(turn, tokenizer)) + 1
            target[cursor:cursor + prompt_length] = IGNORE_INDEX
            cursor += round_length
        if cursor != len(ids):
            raise ValueError("Phi tokenization does not match the Bunny conversation template.")
        input_ids.append(ids[:tokenizer.model_max_length])
        labels.append(target[:tokenizer.model_max_length])
        conversations.append(conversation)
    return dict(input_ids=input_ids, labels=labels, clip_texts=conversations)
