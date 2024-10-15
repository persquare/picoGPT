"""
To run on a single GPU, example:
$ python train.py
"""
import os
import time
import math
from dataclasses import dataclass

from model import GPT
from model import Utils

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
    min_lr: float  = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

@dataclass
class ModelConfig:
    vocab_size: int = TrainConfig.vocab_size
    block_size: int = TrainConfig.block_size
    # model
    n_layer: int = 4
    n_head: int= 4
    n_embd: int = 128
    dropout: float = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False # do we use bias inside LayerNorm and Linear layers?

@dataclass
class HardwareConfig:
    # system HW
    device: str = 'mps' # 'mps' on macbooks or try 'cpu'
    dtype: str = 'float32'
    # various inits, derived attributes, I/O setup
    seed_offset: int = 0
    seed: int = 1337

@dataclass
class SampleConfig:
        start: str = "\n"
        # num_samples: int = 1 # number of samples to draw
        max_new_tokens: int = 500 # number of tokens generated in each sample
        temperature: float = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k: int = 20 # retain only the top_k most likely tokens, clamp others to have 0 probability


class ANN(object):

    def __init__(self, device):
        super().__init__()

        self.device = device
        # init a new model from scratch
        mc = ModelConfig()
        self.model = GPT(mc)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    def _estimate_loss(self, train_data, val_data, tc):
        out = []
        self.model.eval()
        for data in [train_data, val_data]:
            losses = Utils.array_of_zero_tensors(tc.eval_iters)
            for k in range(tc.eval_iters):
                X, Y = Utils.get_batch(data, self.device, tc)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out.append(losses.mean())
        self.model.train()
        return tuple(out)


    def train(self, tc, train_data, val_data):
        self.model.to(self.device)
        optimizer = self.model.configure_optimizers(tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2), self.device)

        # training loop
        X, Y = Utils.get_batch(train_data, self.device, tc) # fetch the very first batch
        t0 = time.time()
        # raw_model = self.model #
        running_mfu = -1.0

        iter_num = 0
        best_val_loss = 1e9

        decay_rate = math.pow(10, math.log10(tc.min_lr/tc.learning_rate)/tc.max_iters)
        lr = tc.learning_rate
        checkpoint = None

        while True:

            # determine and set the learning rate for this iteration
            lr = lr * decay_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % tc.eval_interval == 0:
                (train_loss, val_loss) = self._estimate_loss(train_data, val_data, tc)
                print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = self.model.state_dict()

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(tc.gradient_accumulation_steps):
                logits, loss = self.model(X, Y)
                loss = loss / tc.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = Utils.get_batch(train_data, self.device, tc)
                # backward pass
                loss.backward()
            # clip the gradient
            if tc.grad_clip != 0.0:
                self.model.clip_grad_norm(tc.grad_clip)

            optimizer.step()
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
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, learning rate {100*lr/tc.learning_rate:3.1f}%")
            iter_num += 1

            # termination conditions
            if iter_num > tc.max_iters:
                return checkpoint

    def sample(self, checkpoint, coder, sc):
        device = 'cpu' # override, much faster than hc.device='mps'

        self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.model.to(device)

        start_ids = coder.encode(sc.start)

        y = self.model.generate(start_ids, device, sc)
        return coder.decode(y[0].tolist())


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


def get_data(input_file_path, coder):

    with open(input_file_path, 'r') as f:
        data = f.read()

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = coder.encode(train_data)
    val_ids = coder.encode(val_data)

    return train_ids, val_ids


def main():

    tc = TrainConfig()
    coder = Coder(tc)
    hc = HardwareConfig()
    ann = ANN(hc.device)

    if True:
        input_file_path = os.path.join('data', tc.dataset, 'input.txt')
        train_data, val_data = get_data(input_file_path, coder)
        Utils.set_seed(hc)
        checkpoint = ann.train(tc, train_data, val_data)
        os.makedirs(tc.out_dir, exist_ok=True)
        print(f"saving checkpoint to {tc.out_dir}")
        Utils.save_weights(checkpoint, tc)
        # else:
        # init from a model saved in a specific directory
        checkpoint = Utils.load_weights(tc)
        Utils.set_seed(hc)
        sc = SampleConfig()
        res = ann.sample(checkpoint, coder, sc)
        print(res)

if __name__ == '__main__':
    main()