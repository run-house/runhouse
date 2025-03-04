# Direct translation of TF code from GPT-2 Repo: https://github.com/openai/gpt-2/blob/master/src/model.py
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HParams:
    n_vocab: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


def default_hparams():
    return HParams()


def shape_list(x):
    return list(x.size())


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class LayerNorm(nn.Module):
    def __init__(self, n_state, epsilon=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.epsilon = epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.epsilon)
        return x * self.g + self.b


class Conv1D(nn.Module):
    def __init__(self, nx, nf, w_init_stdev=0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, nx, nf).normal_(std=w_init_stdev))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.bias.size(0),)
        x = torch.matmul(x.view(-1, x.size(-1)), self.weight.squeeze(0).transpose(0, 1))
        x = x + self.bias
        return x.view(*size_out)


class Attention(nn.Module):
    def __init__(self, n_state, hparams):
        super().__init__()
        assert n_state % hparams.n_head == 0
        self.n_head = hparams.n_head
        self.split_size = n_state
        self.n_embd = hparams.n_embd
        self.c_attn = Conv1D(n_state, n_state * 3)
        self.c_proj = Conv1D(n_state, n_state)

    def _attn(self, q, k, v, mask=True):
        w = torch.matmul(q, k.transpose(-1, -2))
        w = w / torch.sqrt(torch.tensor(v.size(-1), dtype=w.dtype))

        if mask:
            # Create attention mask
            nd, ns = q.size(-2), k.size(-2)
            mask = torch.tril(torch.ones(nd, ns, device=q.device)).view(1, 1, nd, ns)
            w = w * mask - 1e10 * (1 - mask)

        w = F.softmax(w, dim=-1)
        return torch.matmul(w, v)

    def _split_heads(self, x):
        new_shape = shape_list(x)[:-1] + [self.n_head, x.size(-1) // self.n_head]
        x = x.view(*new_shape)  # [batch, sequence, n_head, embd_size//n_head]
        return x.permute(0, 2, 1, 3)  # [batch, n_head, sequence, embd_size//n_head]

    def _merge_heads(self, x):
        """Merge heads together over the last dimension."""
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = shape_list(x)[:-2] + [x.size(-2) * x.size(-1)]
        return x.view(*new_shape)

    def forward(self, x, past=None):
        x = self.c_attn(x)
        q, k, v = torch.split(x, self.split_size, dim=2)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        present = torch.stack([k, v], dim=1)

        if past is not None:
            pk, pv = torch.unbind(past, dim=1)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        a = self._attn(q, k, v)
        a = self._merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, hparams):
        super().__init__()
        nx = hparams.n_embd
        self.c_fc = Conv1D(nx, n_state)
        self.c_proj = Conv1D(n_state, nx)

    def forward(self, x):
        h = gelu(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        nx = hparams.n_embd
        self.ln_1 = LayerNorm(nx)
        self.attn = Attention(nx, hparams)
        self.ln_2 = LayerNorm(nx)
        self.mlp = MLP(nx * 4, hparams)

    def forward(self, x, past=None):
        a, present = self.attn(self.ln_1(x), past=past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embd)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embd)
        self.blocks = nn.ModuleList([Block(hparams) for _ in range(hparams.n_layer)])
        self.ln_f = LayerNorm(hparams.n_embd)

    def get_position_ids(self, tokens, past_length):
        batch_size = tokens.size(0)
        nsteps = tokens.size(1)
        positions = past_length + torch.arange(nsteps, device=tokens.device)
        return positions.unsqueeze(0).expand(batch_size, nsteps)

    def forward(self, x, past=None):
        results = {}
        batch_size, sequence_length = x.size()

        past_length = 0 if past is None else past.size(3)
        position_ids = self.get_position_ids(x, past_length)

        h = self.wte(x) + self.wpe(position_ids)

        # Initialize past states if needed
        presents = []
        pasts = (
            torch.unbind(past, dim=1)
            if past is not None
            else [None] * self.hparams.n_layer
        )

        for i, (block, past_state) in enumerate(zip(self.blocks, pasts)):
            h, present = block(h, past=past_state)
            presents.append(present)

        results["present"] = torch.stack(presents, dim=1)
        h = self.ln_f(h)

        # Language model logits
        logits = torch.matmul(h, self.wte.weight.transpose(0, 1))
        results["logits"] = logits

        return results
