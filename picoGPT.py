"""
To run on a single GPU, example:
$ python train.py
"""
import os
import time
import math
import pickle
import contextlib
from dataclasses import dataclass

import torch
import numpy as np

from model import GPT

@dataclass
class TrainConfig:
    out_dir: str = 'out-shakespeare-char'
    eval_interval: int = 250
    log_interval: int = 100
    eval_iters: int = 20
    dataset: str = 'shakespeare_char'
    vocab_size: int = 65
    gradient_accumulation_steps: int = 1 # used to simulate larger batch sizes
    batch_size: int = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 64
    # adamw optimizer
    learning_rate: float  = 1e-3 # max learning rate
    max_iters: int = 8 * eval_interval # total number of training iterations
    weight_decay: float  = 1e-1
    beta1: float  = 0.9
    beta2: float  = 0.95
    grad_clip: float  = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 100 # how many steps to warm up for
    lr_decay_iters: int = 2000 # should be ~= max_iters per Chinchilla
    min_lr: float  = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    # model
    n_layer: int = 4
    n_head: int= 4
    n_embd: int = 128
    dropout: float = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False # do we use bias inside LayerNorm and Linear layers?

@dataclass
class HardwareConfig:
    # system HW
    device: str = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False # use PyTorch 2.0 to compile the model to be faster
    # various inits, derived attributes, I/O setup
    seed_offset: int = 0

class ANN(object):
    def __init__(self):
        super().__init__()
        tc = TrainConfig()
        self.hc = HardwareConfig()
        self.mc = ModelConfig(block_size=tc.block_size, vocab_size=tc.vocab_size)
        torch.manual_seed(1337 + self.hc.seed_offset)
        # init a new model from scratch
        self.model = GPT(self.mc)

    def train(self, tc, train_data, val_data):
        hc = self.hc
        mc = self.mc
        self.model.to(self.hc.device)
        # Cuda only. If enabled=False scaler is a no-op but still needed for convergence?!
        scaler = torch.amp.GradScaler(enabled=False)
        # optimizer
        optimizer = self.model.configure_optimizers(tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2), self.hc.device)

        def get_batch(split):
            # We recreate np.memmap every batch to avoid a memory leak, as per
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            # data_dir = os.path.join('data', tc.dataset)
            if split == 'train':
                # data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
                data = train_data
            else:
                # data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
                data = val_data
            ix = torch.randint(len(data) - mc.block_size, (tc.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+tc.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+tc.block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(hc.device), y.to(hc.device)
            return x, y

        # helps estimate an arbitrarily accurate loss over either split using many batches
        @torch.no_grad()
        def estimate_loss():
            out = {}
            self.model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(tc.eval_iters)
                for k in range(tc.eval_iters):
                    X, Y = get_batch(split)
                    with contextlib.nullcontext():
                        logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            self.model.train() # <-- THIS IS A NO-OP UNLESS IT HAS SERIOUS SIDE EFFECTS
            return out

        # learning rate decay scheduler (cosine with warmup)
        def get_lr(it):
            # 1) linear warmup for warmup_iters steps
            if it < tc.warmup_iters:
                return tc.learning_rate * it / tc.warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > tc.lr_decay_iters:
                return tc.min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - tc.warmup_iters) / (tc.lr_decay_iters - tc.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)

        # training loop
        X, Y = get_batch('train') # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = self.model #
        running_mfu = -1.0

        iter_num = 0
        best_val_loss = 1e9

        checkpoint = None

        while True:

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if tc.decay_lr else tc.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % tc.eval_interval == 0:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if losses['val'] < best_val_loss: # or always_save_checkpoint:
                    best_val_loss = losses['val']
                    checkpoint = raw_model.state_dict()

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(tc.gradient_accumulation_steps):
                with contextlib.nullcontext():
                    logits, loss = self.model(X, Y)
                    loss = loss / tc.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if tc.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % tc.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * tc.gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(tc.batch_size * tc.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > tc.max_iters:
                return checkpoint

    def sample(self, checkpoint, coder):

        device = 'cpu'

        # -----------------------------------------------------------------------------
        start = "Lo!\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        num_samples = 10 # number of samples to draw
        max_new_tokens = 500 # number of tokens generated in each sample
        temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

        seed = 1337
        # -----------------------------------------------------------------------------

        torch.manual_seed(seed + self.hc.seed_offset)
        # ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[hc.dtype]

#        model = GPT(mc)
        self.model.load_state_dict(checkpoint)

        #model.eval()
        self.model.to(device) # Much faster than hc.device='mps'

        start_ids = coder.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        # run generation
        res = []
        with torch.no_grad():
            with contextlib.nullcontext():
                for k in range(num_samples):
                    y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    res.append(coder.decode(y[0].tolist()))

        return "".join(res)

def get_data(input_file_path, coder):

    with open(input_file_path, 'r') as f:
        data = f.read()

    # get all the unique characters that occur in this text
    #chars = sorted(list(set(data)))

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = coder.encode(train_data)
    val_ids = coder.encode(val_data)

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    return train_ids, val_ids

class Coder(object):
    def __init__(self, tc):
        super().__init__()
        input_file_path = os.path.join('data', tc.dataset, 'input.txt')
        with open(input_file_path, 'r') as f:
            data = f.read()
        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s] # encoder: take a string, output a list of integers

    def decode(self, l):
        return ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string

def main():

    tc = TrainConfig()
    coder = Coder(tc)
    ann = ANN()

    if True:
        input_file_path = os.path.join('data', tc.dataset, 'input.txt')
        train_data, val_data = get_data(input_file_path, coder)
        checkpoint = ann.train(tc, train_data, val_data)
        os.makedirs(tc.out_dir, exist_ok=True)
        print(f"saving checkpoint to {tc.out_dir}")
        torch.save(checkpoint, os.path.join(tc.out_dir, 'ckpt.pt'))
    else:
        # init from a model saved in a specific directory
        checkpoint = torch.load(os.path.join(tc.out_dir, 'ckpt.pt'), map_location='cpu', weights_only=True)
        res = ann.sample(checkpoint, coder)
        print(res)

if __name__ == '__main__':
    main()