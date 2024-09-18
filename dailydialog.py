from tqdm import tqdm
from dataclasses import dataclass
import math
import time
import inspect
import os
import numpy as np
import tiktoken
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def save_shard(path, tokens, speaker_tokens):
    tokens_bin = tokens.numpy().tobytes()
    speaker_tokens_bin = speaker_tokens.numpy().tobytes()

    with open(path, "wb") as f:
        # data size
        f.write(np.array(tokens.shape, dtype=np.uint32).tobytes())
        f.write(len(tokens_bin).to_bytes(4, byteorder='little'))
        
        # tokens
        f.write(tokens_bin)
        f.write(speaker_tokens_bin)



def save_dataset_split(filename, dataset_split, enc, shard_size=5e7, endoftext_token=50256, padding_token=50257):
    # os.makedirs(folder_path, exist_ok=True)

    dialogs = []
    speaker_tokens = []
    for dialog_data in dataset_split:
        dialog = list(map(enc.encode, dialog_data["dialog"]))
        dialogs.append(torch.cat(list(map(lambda x:torch.tensor(x + [endoftext_token], dtype=torch.uint16), dialog))))
        speaker_tokens.append(torch.cat(list(map(lambda x:torch.full((len(x[1]) + 1,), x[0] % 2, dtype=torch.uint16), enumerate(dialog)))))
    speaker_tokens = pad_sequence(speaker_tokens, batch_first=True, padding_value=0)
    tokens = pad_sequence(dialogs, batch_first=True, padding_value=padding_token)

    tokens_bin = tokens.numpy().tobytes()
    speaker_tokens_bin = speaker_tokens.numpy().tobytes()

    with open(filename, "wb") as f:
        # data size
        f.write(np.array(tokens.shape, dtype=np.uint32).tobytes())
        f.write(len(tokens_bin).to_bytes(4, byteorder='little'))
        
        # tokens
        f.write(tokens_bin)
        f.write(speaker_tokens_bin)
    

def save_dataset(folder_path, dataset, enc):
    os.makedirs(folder_path, exist_ok=True)

    train_path = os.path.join(folder_path, "train.bin")
    val_path = os.path.join(folder_path, "val.bin")
    test_path = os.path.join(folder_path, "test.bin")

    save_dataset_split(train_path, dataset["train"], enc)
    save_dataset_split(val_path, dataset["validation"], enc)
    save_dataset_split(test_path, dataset["test"], enc)

if __name__ == "__main__":
    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    enc = tiktoken.get_encoding('gpt2')
    save_dataset(r"data\dalydialogs10M", dataset, enc)
    