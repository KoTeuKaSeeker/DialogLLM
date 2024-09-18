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

import warnings
warnings.filterwarnings("ignore")

import os 


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
    def __init__(self, total_batch_size, B, T, model_path, load_model, dataset_path, log_dir, checkpoints_dir):
        self.total_batch_size = total_batch_size
        self.B = B
        self.T = T
        self.model_path = model_path
        self.load_model = load_model
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.checkpoints_dir = checkpoints_dir


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
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1<|endoftext|> token
    n_layers: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

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
        
    
    def forward(self, idx, targets=None):
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

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

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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

def load_tokens(filename, dtype=np.uint16):
    with open(filename, "rb") as f:
        buffer = f.read()
        npt = np.frombuffer(buffer, dtype=dtype)
        ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, path, B, T, process_rank, num_processes, master_process, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = path
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bound, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

def reduce_avg_fn(vals):
    # take average
    return sum(vals) / len(vals)


def run(device_manager: DeviceManager, train_parameters: TrainParameters):
    ddp_rank = device_manager.rank
    ddp_world_size = device_manager.world_size
    master_process = device_manager.master_process
    device = device_manager.device
    print(f"device {device} |")
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    assert train_parameters.total_batch_size % (train_parameters.B * train_parameters.T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = train_parameters.total_batch_size // (train_parameters.B * train_parameters.T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {train_parameters.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
    train_loader = DataLoaderLite(path=train_parameters.dataset_path, B=train_parameters.B, T=train_parameters.T, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process, split="train")
    val_loader = DataLoaderLite(path=train_parameters.dataset_path, B=train_parameters.B, T=train_parameters.T, process_rank=ddp_rank, num_processes=ddp_world_size, master_process=master_process, split="val")
    
    torch.set_float32_matmul_precision('high')
    
    if train_parameters.load_model:
        model, _, _, _ = GPT.from_checkpoint(train_parameters.model_path)
    else:
        model = GPT(GPTConfig(vocab_size=50257))
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation.
    if use_compile:
        model = torch.compile(model) # Тоже не работает
    raw_model = model # always conins the "raw" unwrapped model
    
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = int(375e6) // train_parameters.total_batch_size
    max_steps = int(10e9) // train_parameters.total_batch_size * 2
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min_lerning_rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    
    enc = tiktoken.get_encoding('gpt2')
    
    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_manager=device_manager)
    
    # create the log directory we will write checkpoints to add log to
    log_dir = train_parameters.log_dir
    log_file = os.path.join(log_dir, f"log.txt")
    if master_process:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "w") as file: # open for writing to clear the file
            pass
    
    checkpoints_dir = train_parameters.checkpoints_dir
    if master_process:
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    step = 0
    while step < max_steps:
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if  step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 1
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    # with torch.autocast(device_type=device, dtype=torch.bfloat16): # Почему-то не ускоряет вычисления, а замедляет
                    #     logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                val_loss_accum = device_manager.all_reduce(val_loss_accum) / ddp_world_size
                device_manager.mark_step()
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        
            if (step % 5000 == 0 or last_step):
                if master_process:
                    print(f"Saving the model model_{step:05d}.pt...")
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(checkpoints_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(checkpoint, checkpoint_path)
                device_manager.mark_step()
                device_manager.rendezvous("checkpoint_sync_point")
            

        # once in a while generate from the model (except step 0, which is noise)
        if ((step % 250 == 0) or last_step) and (not use_compile):
            device_manager.mark_step()
            device_manager.rendezvous("generation_sync_point")
            device_manager.master_print(f"Generating samples from model...")
            model.eval()
            num_return_sequences = 1
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            rng_state = torch.get_rng_state()
            torch.manual_seed(123 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    logits, _ = model(xgen) # (B, T, vocab_size)
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
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")
            torch.set_rng_state(rng_state)

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for microstep in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # with torch.autocast(device_type=device, dtype=torch.bfloat16): # Почему-то не ускоряет вычисления, а замедляет
            #     logits, loss = model(x, y)
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        loss_accum = device_manager.all_reduce(loss_accum) / ddp_world_size
        device_manager.mark_step()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        device_manager.optimizer_step(optimizer)
        t1 = time.time()
        dt = (t1 - t0) * 1000 # in millisecondss
        token_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_second = token_processed / (t1 - t0) # tok/s
        if master_process:
            print(f"step {step} | loss: {loss_accum.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second}")
        step += 1


if __name__ == "__main__":
    rank = 0
    world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_batch_size = 524288 # count number of tokens
    B = 16 # micro batch size
    T = 1024 # sequence length
    model_path = r"models\casual-model\v3\model.pt"
    load_model = False

    dataset_path = r"data\fineweb"
    log_dir = r"log"
    checkpoints_dir = r"checkpoints"

    device_manager = DeviceManager(rank, world_size, master_process, device)
    train_parameters = TrainParameters(total_batch_size, B, T, model_path, load_model, dataset_path, log_dir, checkpoints_dir)

    run(device_manager, train_parameters)