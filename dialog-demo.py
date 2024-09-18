from tqdm import tqdm
from dataclasses import dataclass
import math
import time
import inspect
import os
import numpy as np
import tiktoken

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import kagglehub
import shutil

import warnings
warnings.filterwarnings("ignore")

class DeviceManager():
    def __init__(self, rank, world_size, master_process, device):
        self.rank = rank
        self.world_size = world_size
        self.master_process = master_process
        self.device = device
    
    def master_print(self, str):
        if self.master_process:
            print(str)
    
    def rendezvous(self, str):
        pass

    def mark_step(self):
        torch.cuda.synchronize()
    
    def optimizer_step(self, optimizer):
        optimizer.step()
    
    def all_reduce(self, tensor):
        return tensor
    
    def save(self, checkpoint, filename):
        torch.save(checkpoint, filename)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all head, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn  = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1<|endoftext|> token + 1<|padding_token|> token
    n_layers: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    padding_token: int = 50255 # padding token will be ignored by a loss function
    endoftext_token: int = 50256 # end of text token

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        
    
    def forward(self, idx, targets=None, padding_token=None):
        if padding_token is None:
            padding_token = self.config.padding_token
        
        padding_token = torch.tensor(padding_token, dtype=torch.long, device=idx.device)
        
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size must be <= {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=padding_token)
        return logits, loss
    
    def generate(self, idx, stop_text, enc, device_manager: DeviceManager, max_generations=256, context_window_size=1024, batch_check_size=10):    
        B, T = idx.shape
        endoftext_token = torch.tensor([self.config.endoftext_token], dtype=torch.long, device=idx.device)
        count_generations = 0
        stop_token_id = 0
        count_stop_tokens = 0
        while count_generations < max_generations:
            # forward the model to get the logits
            with torch.no_grad():
                device_manager.mark_step()
                idx_slice = idx if idx.size(1) < context_window_size else idx[-context_window_size:]
                logits, _ = self(idx_slice) # (B, T, vocab_size)
                # take the logits at hte last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilites
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the tok-k probabilities
                ix = torch.multinomial(topk_probs, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)    
                # append to the sequence
                idx = torch.cat((idx, xcol), dim=1)
                device_manager.mark_step()
                
                if B == 1:
                    gen_token = xcol[0, 0].item()
                    token_text = enc.decode([gen_token]) if 0 <= gen_token < 50257 else ""
                    count_stop_tokens += 1
                    for ch in token_text:
                        if ch == stop_text[stop_token_id]:
                            stop_token_id += 1
                            if stop_token_id >= len(stop_text):
                                count_generations += 1
                                break
                        else:
                            stop_token_id = 0
                            count_stop_tokens = 0
                    if stop_token_id >= len(stop_text):
                        break
            count_generations += 1
        return idx[:, -count_generations:-count_stop_tokens]
        

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weigts from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # create a from scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # remove tril matrix from keys

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = list(filter(lambda k: not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias'), sd_keys_hf))
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weight we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape # check for square shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanila copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = GPT(checkpoint["config"])
        
        sd = model.state_dict()
        sd_l = checkpoint["model"]
        with torch.no_grad():
            for key in sd.keys():
                if key in sd_l:
                    sd[key].copy_(sd_l[key])
        
        step = checkpoint['step']
        epoch = checkpoint['epoch'] if "epoch" in checkpoint else 0 
        
        optimizer_sd = checkpoint['optimizer']
        return model, optimizer_sd, step, epoch
    
    @classmethod
    def from_gen_model(cls, gen_model):
        config = GPTConfig(vocab_size=gen_model.config.vocab_size+1)
        model = GPT(config)
        sd = model.state_dict()
        sd_l = gen_model.state_dict()
        
        padded_keys = ["transformer.wte.weight", "lm_head.weight"]
        wse_key = "transformer.wse.weight"
        
        for key in sd.keys():
            if not any(key.endswith(k) for k in padded_keys):
                if not key.endswith(wse_key):
                    with torch.no_grad():
                        sd[key].copy_(sd_l[key])
            else:
                with torch.no_grad():
                    assert sd[key].shape > sd_l[key].shape, "Padding token has not been added"
                    sd[key][:sd_l[key].shape[0]] = sd_l[key]
        
        sd["lm_head.weight"][-1] = torch.rand_like(sd["lm_head.weight"][-1], device=sd["lm_head.weight"].device) * 0.0000

        return model
        

    def configure_optimizers(self, weight_decay, learning_rate, device_manager: DeviceManager):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayd, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        device_manager.master_print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        device_manager.master_print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        # create AdamW optimizer and use the fused verssion if it is available
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=False)
        return optimizer


def load_model(filename):
    if not os.path.exists(filename):
        folder_path = os.path.dirname(filename)
        os.makedirs(folder_path, exist_ok=True)

        path = kagglehub.model_download("danildolgov/nanogpt/pyTorch/dialog-model")
        cache_file = os.listdir(path)[0]
        
        shutil.move(os.path.join(path, cache_file), filename)
    
    return filename


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    device_manager = DeviceManager(0, 1, True, device)
    
    B = 4 # micro batch size
    T = 1024 # sequence length

    enc = tiktoken.get_encoding('gpt2')

    torch.set_float32_matmul_precision('high')

    model, _, _, _ = GPT.from_checkpoint(load_model(r"models\dialog-model\v3\model.pt"))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    
    start_speaker_text = "[SPEAKER-A]:"
    start_bot_text = "[SPEAKER-B]:"

    start_speaker_tokens = enc.encode(start_speaker_text)
    start_bot_tokens = enc.encode(start_bot_text)

    model.eval()
    idx = torch.empty((1, 0), dtype=torch.long, device=device)
    sp_x = torch.empty((1, 0), dtype=torch.long, device=device)
    while True:
        statement = input("[SPEAKER]: ")
        statement_x = torch.tensor(start_speaker_tokens + enc.encode(statement) + start_bot_tokens, dtype=torch.long, device=device).unsqueeze(0)

        idx = torch.cat((idx, statement_x), dim=1)
        
        gen_x = model.generate(idx, start_speaker_text, enc, device_manager, max_generations=60, context_window_size=T)
        
        idx = torch.cat((idx, gen_x), dim=1)

        gen_statement = enc.decode(gen_x[0].tolist())
        print(f"[BOT]: {gen_statement}")
