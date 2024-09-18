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

import warnings
warnings.filterwarnings("ignore")

import os 

# help code
#############################################################################
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
        

class TrainParameters():
    def __init__(self, total_batch_size, B, count_epoch, count_validation_steps, validation_freq, generation_freq, save_freq, model_path, generation_statements):
        self.total_batch_size = total_batch_size
        self.B = B
        self.count_epoch = count_epoch
        self.count_validation_steps = count_validation_steps
        self.validation_freq = validation_freq
        self.generation_freq = generation_freq
        self.save_freq = save_freq
        self.model_path = model_path
        self.generation_statements = generation_statements


class LearningRateScheduler():
    def __init__(self, min_lr, max_lr, warmup_steps_portion):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps_portion = warmup_steps_portion
    
    def init(self, train_parameters: TrainParameters, dataset_len):
        count_steps_in_epoch = dataset_len // train_parameters.total_batch_size
        self.max_steps = count_steps_in_epoch * train_parameters.count_epoch
        self.warmup_steps = int(self.max_steps * self.warmup_steps_portion)
    
    def get_lr(self, total_iteration):
        # 1) linear warmup for warmup_iters steps
        if total_iteration < self.warmup_steps:
            return self.max_lr * (total_iteration + 1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min_lerning_rate
        if total_iteration > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (total_iteration - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class Saver():
    def __init__(self, log_dir, checkpoints_dir, device_manager: DeviceManager):
        self.log_dir = log_dir
        self.checkpoints_dir = checkpoints_dir
        self.device_manager = device_manager
        self.log_file = os.path.join(log_dir, f"log.txt")
        self.best_val_loss = None

        if device_manager.master_process:
            os.makedirs(log_dir, exist_ok=True)
            with open(self.log_file, "w") as file: # open for writing to clear the file
                pass
            
            os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def save_log(self, line: str):
        pass
    
    def save_checkpoint(self, filename, model, optimizer, step, epoch, val_loss):
        if self.device_manager.master_process:
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step,
                'epoch': epoch,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict()
            }
            self.device_manager.save(checkpoint, filename)

        
    def save_last_and_best_checkpoint(self, model, optimizer, step, epoch, val_loss):
        self.device_manager.rendezvous("s_save_point")
        self.device_manager.mark_step()
        
        if self.device_manager.master_process:
            last_model_path = os.path.join(self.checkpoints_dir, "last.pt")
            self.save_checkpoint(last_model_path, model, optimizer, step, epoch, val_loss)
            
            if self.best_val_loss is None or val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(self.checkpoints_dir, "best.pt")
                self.save_checkpoint(best_model_path, model, optimizer, step, epoch, val_loss)
        
        self.device_manager.rendezvous("save_point")
        self.device_manager.mark_step()


# model difinition
#############################################################################
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
                    for ch in token_text:
                        if ch == stop_text[stop_token_id]:
                            stop_token_id += 1
                            if stop_token_id >= len(stop_text):
                                count_generations += 1
                                break
                        else:
                            stop_token_id = 0
                    if stop_token_id >= len(stop_text):
                        break
            count_generations += 1
        return idx[:, -count_generations:]
        

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

# data loaders
#############################################################################
class DailyDialogDataset(IterableDataset):
    def __init__(self, batch_size: int, rank: int, world_size: int, split: str, enc):
        self.batch_size = batch_size
        dataset = load_dataset("daily_dialog", trust_remote_code=True)
        raw_dialogs = dataset['train'] if split == 'train' else dataset['validation']
        
        self.speaker_text = "[SPEAKER-A]:"
        self.bot_text = "[SPEAKER-B]:"
        self.speaker_tokens = enc.encode(self.speaker_text)
        self.bot_tokens = enc.encode(self.bot_text)
        id_tokens = [self.speaker_tokens, self.bot_tokens]
        end_of_text_token = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
        padding_token = 50255
        
        dialogs = []
        for dialog_data in raw_dialogs:
            dialog = list(map(enc.encode, dialog_data["dialog"]))
            dialogs.append(torch.cat(list(map(lambda x:torch.tensor(id_tokens[x[0] % 2] + x[1], dtype=torch.long), enumerate(dialog)))))
        self.tokens = pad_sequence(dialogs, batch_first=True, padding_value=padding_token)

        self.tokens = self.tokens[..., :128] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.current_index = rank * batch_size
        
        self.rank = rank
        self.world_size = world_size
        self.T = self.tokens.size(1)
        
    def __iter__(self):
        while self.current_index + self.batch_size <= self.tokens.size(0):
            buffer = self.tokens[self.current_index:self.current_index + self.batch_size]
            x = buffer[..., :-1]
            y = buffer[..., 1:]
            yield x, y
            self.current_index += self.world_size * self.batch_size
        self.current_index = self.rank * self.batch_size

    def __len__(self):
        return self.tokens.shape[0]

#############################################################################
def accum_train_step(model, optimizer, dataset_iterator, device_manager: DeviceManager, grad_accum_steps):
    optimizer.zero_grad()
    loss_accum = torch.tensor(0.0, dtype=torch.float, device=device_manager.device)
    grad_accum_steps_tensor = torch.tensor(grad_accum_steps, dtype=torch.float, device=device_manager.device)
    for step in range(grad_accum_steps):
        try:
            tokens_x_batch, tokens_y_batch = next(dataset_iterator)
            tokens_x_batch = tokens_x_batch.to(device_manager.device)
            tokens_y_batch = tokens_y_batch.to(device_manager.device)
        except StopIteration:
            device_manager.mark_step()
            loss_accum = loss_accum.cpu()
            return True, loss_accum, loss_accum

        device_manager.mark_step()
        logits, loss = model(tokens_x_batch, tokens_y_batch)
        loss = loss / grad_accum_steps_tensor
        loss.backward()
        device_manager.mark_step()

        loss_accum += loss.detach()

    loss_accum = device_manager.all_reduce(loss_accum) / device_manager.world_size
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    device_manager.optimizer_step(optimizer)
    
    loss_accum = loss_accum.cpu()
    norm = norm.cpu()
    return False, loss_accum, norm


def get_validation_loss(model, train_parameters: TrainParameters, validation_dataset: DailyDialogDataset, device_manager: DeviceManager):
    count_validation_steps_tensor = torch.tensor(train_parameters.count_validation_steps, dtype=torch.float, device=device_manager.device)
    with torch.no_grad():
        val_loss = torch.tensor(0.0, dtype=torch.float, device=device_manager.device)
        validation_iterator = iter(validation_dataset)
        for step in range(train_parameters.count_validation_steps):
            try:
                tokens_x_batch, tokens_y_batch = next(validation_iterator)
                tokens_x_batch = tokens_x_batch.to(device_manager.device)
                tokens_y_batch = tokens_y_batch.to(device_manager.device)
            except StopIteration:
                break

            device_manager.mark_step()
            logits, loss = model(tokens_x_batch, tokens_y_batch)
            loss = loss / count_validation_steps_tensor
            device_manager.mark_step()

            val_loss += loss.detach()
        val_loss = device_manager.all_reduce(val_loss) / device_manager.world_size
        val_loss = val_loss.cpu()
    device_manager.mark_step()
    return val_loss

def generate_samples(model, statements, enc, device_manager: DeviceManager, train_dataset: DailyDialogDataset, max_generations=64, context_window_size=1024):
    #statements_tensors = [torch.tensor(enc.encode(statement) + [model.config.endoftext_token], dtype=torch.long, device=device_manager.device).unsqueeze(0) for statement in statements]
    statements_tensors = [torch.tensor(train_dataset.speaker_tokens + enc.encode(statement) + train_dataset.bot_tokens, dtype=torch.long, device=device_manager.device).unsqueeze(0) for statement in statements]

    model.eval()
    gen_statements_x = []
    idx = torch.empty((1, 0), dtype=torch.long, device=device_manager.device)
    for statement_x in statements_tensors:
        device_manager.mark_step()
        idx = torch.cat((idx, statement_x), dim=1)
        
        gen_x = model.generate(idx, train_dataset.speaker_text, enc, device_manager, max_generations=max_generations, context_window_size=context_window_size)
        
        idx = torch.cat((idx, gen_x), dim=1)
        device_manager.mark_step()
        
        gen_statements_x.append(gen_x)
    
    gen_statements = [enc.decode(list(filter(lambda x: 0 <= x < 50255, statement[0].cpu().tolist()))) for statement in gen_statements_x]
    for statement, gen_statement in zip(statements, gen_statements):
        device_manager.master_print(f"[SPEAKER]: {statement}")
        device_manager.master_print(f"[BOT]: {gen_statement}")
    
    device_manager.rendezvous("gen_sync")

def print_log(total_iteration, epoch, step, loss, val_loss, learning_rate, norm, dt, grad_accum_steps, 
              device_manager: DeviceManager, train_parameters: TrainParameters, train_dataset: DailyDialogDataset):
    token_processed = train_parameters.B * train_dataset.T * grad_accum_steps * device_manager.world_size
    tokens_per_second = token_processed / dt
    val_loss_str = str(f"{val_loss.item():.6f}") if val_loss is not None else "-"
    device_manager.master_print(f"total_step {total_iteration}, epoch {epoch}, step {step} | loss: {loss.item():.6f} | val_loss: {val_loss_str} | lr {learning_rate:.4e} | norm: {dt:.4f} | dt: {(dt * 1000.0):.2f}ms | tok/sec: {tokens_per_second}")

    
def save_model(saver: Saver, model, optimizer, step, epoch, train_parameters: TrainParameters, device_manager: DeviceManager, val_loss):
    device_manager.master_print("Saving model...")
    saver.save_last_and_best_checkpoint(model, optimizer, step, epoch, val_loss)
    

def run(device_manager: DeviceManager, train_parameters: TrainParameters, learning_rate_scheduler: LearningRateScheduler, saver: Saver):
    print(f"device: {device_manager.device} |")

    assert train_parameters.total_batch_size % (train_parameters.B * device_manager.world_size) == 0, "make sure total_batch_size is divisible by B * world_size"
    grad_accum_steps = train_parameters.total_batch_size // (train_parameters.B * device_manager.world_size)
    device_manager.master_print(f"total desired batch size: {train_parameters.B}")
    device_manager.master_print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    enc = tiktoken.get_encoding('gpt2')

    train_dataset = DailyDialogDataset(train_parameters.B, device_manager.rank, device_manager.world_size, "train", enc)
    val_dataset = DailyDialogDataset(train_parameters.B, device_manager.rank, device_manager.world_size, "val", enc)
    
    config = GPTConfig(vocab_size=50257)
    model, _, _, _ = GPT.from_checkpoint(train_parameters.model_path)
    model.to(device_manager.device)

    generate_samples(model, train_parameters.generation_statements, enc, device_manager, train_dataset, max_generations=64, context_window_size=1024)

    l_epoch = 0
    l_step = 0

    
    train_dataset.current_index += l_step * train_dataset.batch_size * train_dataset.world_size
    
    learning_rate_scheduler.init(train_parameters, len(train_dataset))

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate_scheduler.max_lr, device_manager=device_manager)
    # optimizer.load_state_dict(optimizer_sd)

    total_iteration = l_epoch * (len(train_dataset) // train_parameters.total_batch_size) + l_step
    for epoch in range(l_epoch, train_parameters.count_epoch + l_epoch):
        data_exhausted = False
        dataset_iterator = iter(train_dataset)
        step = l_step if epoch == 0 else 0
        t0 = time.time()
        while not data_exhausted:
            model.train()
            data_exhausted, loss_accum, norm = accum_train_step(model, optimizer, dataset_iterator, device_manager, grad_accum_steps)

            if not data_exhausted:
                learning_rate = learning_rate_scheduler.get_lr(total_iteration)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                
                model.eval()
                
                is_val_iteration = total_iteration % train_parameters.validation_freq == 0
                is_save_iteration = total_iteration % train_parameters.save_freq == 0
                is_generate_iteration = total_iteration % train_parameters.generation_freq == 0
                
                val_loss = None
                if is_val_iteration or is_save_iteration:
                    val_loss = get_validation_loss(model, train_parameters, val_dataset, device_manager)
                
                if is_save_iteration:
                    save_model(saver, model, optimizer, step, epoch, train_parameters, device_manager, val_loss.item())

                if is_generate_iteration:
                    generate_samples(model, train_parameters.generation_statements, enc, device_manager, train_dataset, max_generations=64, context_window_size=1024)
                
                t1 = time.time()
                dt = t1 - t0
                print_log(total_iteration, epoch, step, loss_accum, val_loss, learning_rate, norm, dt, grad_accum_steps, device_manager, train_parameters, train_dataset)
                

                t0 = time.time()
                
                total_iteration += 1
                step += 1
    

if __name__ == "__main__":
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    torch.set_float32_matmul_precision('high')

    rank = 0
    world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_batch_size = 4
    B = 1
    count_epoch = 150
    count_validation_steps = 5
    validation_freq = 30
    generation_freq = 30
    save_freq = 30
    model_path = r"models\dialog-model\best_last_model.pt"
    generation_statements = [
        "Say, Jim, how about going for a few beers after dinner?",
        "What do you mean? It will help us to relax.",
        "I guess you are right. But what shall we do? I don't feel like sitting at home.",
        "That's a good idea. I hear Mary and Sally often go there to play pingpong. Perhaps we can make a foursome with them.",
        "Good. Let's go now."
    ]

    max_learning_rate = 6e-4
    min_learning_rate = max_learning_rate * 0.1
    warmup_steps_portion = 0.0375

    log_dir = "log/"
    checkpoints_dir = "checkpoints/finetune-checkpoints"

    device_manager = DeviceManager(rank, world_size, master_process, device)
    train_parameters = TrainParameters(total_batch_size, B, count_epoch, count_validation_steps, validation_freq, generation_freq, save_freq, model_path, generation_statements)
    learning_rate_scheduler = LearningRateScheduler(min_learning_rate, max_learning_rate, warmup_steps_portion)
    saver = Saver(log_dir, checkpoints_dir, device_manager)

    run(device_manager, train_parameters, learning_rate_scheduler, saver)

