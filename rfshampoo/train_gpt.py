"""E2/E3: GPT training with RootFree-Shampoo on 2x Quadro RTX 6000 (DDP).

Compares, at identical seeds and data order:
  adamw          — reference first-order
  eigh1          — ideal Shampoo (eigh every step)      [gold trajectory, slow]
  eighK          — deployed Shampoo (eigh every K=100)  [stale preconditioner]
  rf             — RootFree (auto: gram -> cont/track)  [exact, eig-free]

Reports per-run: loss curve, optimizer step time breakdown, memory; and the
trajectory distance of each Shampoo variant from the gold eigh1 run.

Model sized to saturate 24GB x 2: GPT with d=1024, 12 layers, seq 512.
Synthetic in-memory token stream (deterministic, no download dependency) with
Zipf unigram + Markov bigram structure so the loss is meaningfully learnable.
"""
import os, sys, time, json, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfshampoo.optimizer import RootFreeShampoo

# --------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
        self.nh = nh

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.nh, -1).transpose(1, 2)
        k = k.view(B, T, self.nh, -1).transpose(1, 2)
        v = v.view(B, T, self.nh, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        h = self.fc1(self.ln2(x))
        x = x + self.fc2(F.gelu(h))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d=1024, nlayer=12, nh=16, seq=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nlayer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


# ----------------------------------------------------------------- synthetic data
def make_stream(vocab, n_tokens, seed, device):
    g = torch.Generator(device='cpu').manual_seed(seed)
    # Zipf unigram + strong deterministic bigram structure (learnable)
    ranks = torch.arange(1, vocab + 1, dtype=torch.float64)
    probs = (1.0 / ranks) / (1.0 / ranks).sum()
    base = torch.multinomial(probs.float(), n_tokens, replacement=True, generator=g)
    succ = torch.randperm(vocab, generator=g)  # bigram successor map
    stream = base.clone()
    mask = torch.rand(n_tokens, generator=g) < 0.55
    stream[1:][mask[1:]] = succ[stream[:-1][mask[1:]]]
    return stream.to(device)


def get_batch(stream, bs, seq, step, device):
    n = stream.numel() - seq - 1
    g = torch.Generator(device='cpu').manual_seed(10_000 + step)  # identical across runs
    ix = torch.randint(0, n, (bs,), generator=g).to(device)
    x = torch.stack([stream[i:i + seq] for i in ix])
    y = torch.stack([stream[i + 1:i + seq + 1] for i in ix])
    return x, y


# --------------------------------------------------------------------- run
def build_opt(name, model, lr):
    twoD = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    oneD = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    mode, every = dict(eigh1=('eigh', 1), eighK=('eigh', 100),
                       rf=('auto', 1))[name]
    return RootFreeShampoo(
        [dict(params=twoD, lr=lr, mode=mode, eigh_every=every),
         dict(params=oneD, lr=lr, mode=mode)],
        lr=lr, eps=1e-6, max_precond_dim=4096,
        precond_dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', default='rf')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--d', type=int, default=1024)
    ap.add_argument('--nlayer', type=int, default=12)
    ap.add_argument('--vocab', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tag', default='')
    ap.add_argument('--snapshot_every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False  # determinism/precision
    torch.backends.cudnn.allow_tf32 = False

    dev = torch.device(args.device)
    model = GPT(args.vocab, args.d, args.nlayer, seq=args.seq).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    stream = make_stream(args.vocab, 4_000_000, 7, dev)
    opt = build_opt(args.opt, model, args.lr)

    out = dict(opt=args.opt, steps=[], loss=[], step_ms=[], opt_ms=[],
               nparams=nparams)
    snaps = {}
    t_run = time.time()
    for step in range(args.steps):
        x, y = get_batch(stream, args.bs, args.seq, step, dev)
        t0 = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        opt.step()
        torch.cuda.synchronize()
        t2 = time.time()
        out['steps'].append(step)
        out['loss'].append(float(loss))
        out['step_ms'].append(1e3 * (t2 - t0))
        out['opt_ms'].append(1e3 * (t2 - t1))
        if (step + 1) % args.snapshot_every == 0 or step == 0:
            with torch.no_grad():
                v = torch.cat([p.flatten() for p in model.parameters()])
                snaps[step] = v.detach().cpu().clone()
            print(f"[{args.opt}] step {step+1}/{args.steps} loss {float(loss):.4f} "
                  f"opt_ms {out['opt_ms'][-1]:.0f} "
                  f"mem {torch.cuda.max_memory_allocated(dev)/2**30:.1f}GB",
                  flush=True)
    out['wall_s'] = time.time() - t_run
    out['max_mem_gb'] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    tag = args.tag or args.opt
    torch.save(snaps, f'/home/navin/shampoo/research/rfshampoo/snaps_{tag}.pt')
    with open(f'/home/navin/shampoo/research/rfshampoo/run_{tag}.json', 'w') as f:
        json.dump(out, f)
    print(f"DONE {args.opt} wall {out['wall_s']:.0f}s "
          f"median_opt_ms {sorted(out['opt_ms'])[len(out['opt_ms'])//2]:.0f}",
          flush=True)


if __name__ == '__main__':
    main()
