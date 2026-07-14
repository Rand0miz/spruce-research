# Multi-Prototype MaxSim Leaf Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mean-pooled block features with multi-prototype (mean + outliers) MaxSim features in teacher extraction and the flat gate, so a single-token needle survives block summarization.

**Architecture:** Each 64-token block is summarized by P=8 prototype vectors (proto 0 = block mean, protos 1..7 = the tokens farthest in L2 from that mean) on BOTH the query and key side. The gate scores a block pair by MaxSim — the max dot over all query-proto × key-proto pairs — computed as a running max looped over key prototypes to bound memory. Only feature production changes; the raw `scores [L,G,qb,kb]` tensor and everything downstream (`loss.py`, `recall.py`, `train.py`, `eval_gate.py`, `interfaces/`) are untouched.

**Tech Stack:** Python 3.12, PyTorch, transformers (Qwen2.5-Coder). No pytest in this repo — tests are standalone scripts run with `python scripts/test_*.py`, matching `scripts/test_validator.py`; each asserts and exits nonzero on failure.

## Global Constraints

- Block size = 64 (unchanged). Prototypes per block P = 8 (1 mean + 7 outliers), both sides.
- Prototype selection uses Q/K vectors ALONE (inference-reproducible) — never teacher mass.
- Target tensor `pooled [1, L, H, qb, kb]` is NEVER changed — it stays the exact chunked attention mass.
- Must fit 8GB VRAM: no `[L,G,qb,kb,P,P]` materialization; MaxSim uses a running max over the P key-prototypes.
- FlatGate parameter shape/count unchanged: per-layer `Wq, Wk` of shape `[L, d, p]`.
- Extraction and retraining are run by the USER, not this plan. The plan lands code only. Do not run `scripts/extract_teacher_targets.py` or `selector/train.py`.
- Downstream files stay untouched: `selector/loss.py`, `selector/recall.py`, `selector/train.py`, `scripts/eval_gate.py`, `interfaces/*`.

---

### Task 1: `_proto_pool_seq` — prototype pooling primitive

**Files:**
- Modify: `teacher/chunked_extract.py` (add `_proto_pool_seq` next to `_mean_pool_seq`)
- Test: `scripts/test_proto_pool.py` (create)

**Interfaces:**
- Consumes: nothing (leaf primitive).
- Produces: `_proto_pool_seq(x, block, P) -> Tensor` where `x` is `[b, H, L, d]` and the return is `[b, H, nb, P, d]` with `nb = ceil(L/block)`. Proto index 0 is the block mean over real (non-pad) rows; protos 1..P-1 are the P-1 tokens farthest in squared-L2 from that mean; padded/short slots are filled with the mean.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_proto_pool.py`:

```python
"""Standalone test for _proto_pool_seq. Run: python scripts/test_proto_pool.py"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from teacher.chunked_extract import _proto_pool_seq


def test_shape_and_mean():
    b, H, L, d, block, P = 1, 2, 128, 4, 64, 8
    x = torch.randn(b, H, L, d)
    out = _proto_pool_seq(x, block, P)
    assert out.shape == (b, H, 2, P, d), out.shape
    # proto 0 == full-block mean for a full (non-ragged) block
    mean0 = x[:, :, :block, :].mean(dim=2)
    assert torch.allclose(out[:, :, 0, 0, :], mean0, atol=1e-5)


def test_outlier_is_kept():
    # one token planted far from the rest -> must appear among protos 1..P-1
    b, H, d, block, P = 1, 1, 4, 64, 8
    x = torch.randn(b, H, block, d) * 0.01          # tight cluster
    x[0, 0, 37, :] = torch.tensor([50.0, 50.0, 50.0, 50.0])   # the needle
    out = _proto_pool_seq(x, block, P)              # [1,1,1,P,d]
    protos = out[0, 0, 0]                           # [P, d]
    needle = x[0, 0, 37]
    hit = (protos - needle).pow(2).sum(-1).min().item()
    assert hit < 1e-6, f"needle not preserved, min dist^2={hit}"


def test_short_block_pads_with_mean():
    # ragged last block with fewer real rows than P-1 outliers
    b, H, d, block, P = 1, 1, 4, 64, 8
    L = 64 + 3                                       # last block has 3 real rows
    x = torch.randn(b, H, L, d)
    out = _proto_pool_seq(x, block, P)              # [1,1,2,P,d]
    last = out[0, 0, 1]                             # [P,d]  last block
    real_mean = x[0, 0, 64:, :].mean(dim=0)
    assert torch.allclose(last[0], real_mean, atol=1e-5)   # proto 0 = real-row mean
    # no proto is a zero-pad row (all pad rows were masked out of selection)
    assert not torch.allclose(last, torch.zeros_like(last))


if __name__ == "__main__":
    test_shape_and_mean()
    test_outlier_is_kept()
    test_short_block_pads_with_mean()
    print("OK test_proto_pool")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/test_proto_pool.py`
Expected: FAIL with `ImportError: cannot import name '_proto_pool_seq'`

- [ ] **Step 3: Write minimal implementation**

In `teacher/chunked_extract.py`, add after `_mean_pool_seq` (around line 37):

```python
def _proto_pool_seq(x, block, P):
    """[b,H,L,d] -> [b,H,nb,P,d]: P prototype vectors per block.
    proto 0 = block mean (real rows only); protos 1..P-1 = the P-1 tokens
    farthest (squared-L2) from that mean. Short/ragged blocks (fewer than P-1
    real outlier rows) fill spare slots with the mean, so max-pooling over
    prototypes is unaffected. Selection uses only x -> reproducible at inference."""
    b, H, L, d = x.shape
    nb = (L + block - 1) // block
    pad = nb * block - L
    if pad:
        x = F.pad(x, (0, 0, 0, pad))                 # zero-pad ragged tail
    xb = x.view(b, H, nb, block, d)                  # [b,H,nb,block,d]

    counts = xb.new_full((nb,), float(block))
    if pad:
        counts[-1] = block - pad                     # real rows in last block
    valid = (torch.arange(block, device=x.device)[None, None, None, :]
             < counts.view(1, 1, nb, 1))             # [1,1,nb,block] bool
    vf = valid.to(xb.dtype)[..., None]               # [1,1,nb,block,1]
    mean = (xb * vf).sum(dim=3) / counts.view(1, 1, nb, 1)   # [b,H,nb,d]

    dist = (xb - mean[:, :, :, None, :]).pow(2).sum(dim=-1)  # [b,H,nb,block]
    dist = dist.masked_fill(~valid, float("-inf"))   # never pick a pad row
    k = min(P - 1, block)
    idx = dist.topk(k, dim=-1).indices               # [b,H,nb,k]
    gidx = idx[..., None].expand(-1, -1, -1, -1, d)
    outliers = xb.gather(3, gidx)                     # [b,H,nb,k,d]
    # if a block had fewer than k real rows, some picked rows are pad (dist=-inf):
    # replace those with the mean.
    sel_dist = dist.gather(-1, idx)                   # [b,H,nb,k]
    bad = torch.isinf(sel_dist)[..., None]            # [b,H,nb,k,1]
    outliers = torch.where(bad, mean[:, :, :, None, :].expand_as(outliers), outliers)

    protos = torch.cat([mean[:, :, :, None, :], outliers], dim=3)   # [b,H,nb,1+k,d]
    if 1 + k < P:                                     # only if block < P-1 (not at 64/8)
        extra = mean[:, :, :, None, :].expand(-1, -1, -1, P - (1 + k), -1)
        protos = torch.cat([protos, extra], dim=3)
    return protos                                     # [b,H,nb,P,d]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/test_proto_pool.py`
Expected: `OK test_proto_pool`

- [ ] **Step 5: Commit**

```bash
git add teacher/chunked_extract.py scripts/test_proto_pool.py
git commit -m "Add _proto_pool_seq: mean + outlier prototypes per block"
```

---

### Task 2: Wire proto features into capture + extraction entry

**Files:**
- Modify: `teacher/chunked_extract.py` (`_capture_attention`, `get_pooled_targets`, module `_PROTO`)
- Modify: `scripts/extract_teacher_targets.py` (save dict: `proto` field + shape comments)
- Test: `scripts/test_group_avg.py` (create)

**Interfaces:**
- Consumes: `_proto_pool_seq(x, block, P)` from Task 1.
- Produces:
  - `_group_avg_tokens(x, G) -> Tensor`: `[b, H, L, d] -> [b, G, L, d]`, averaging the `H/G` query heads within each kv-group per token.
  - `get_pooled_targets(...)` now returns `pooledQ_stack [1, L, G, qb, P, d]` and `pooledK_stack [1, L, G, kb, P, d]` (was `[1,L,H,qb,d]` / `[1,L,kv,kb,d]`); `pooled_stack [1,L,H,qb,kb]` and `seq_len` unchanged. Module constant `_PROTO = 8`.
  - `scripts/extract_teacher_targets.py` saved `.pt` gains key `"proto": <P>`; `pooledQ`/`pooledK` are 6-D.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_group_avg.py`:

```python
"""Standalone test for _group_avg_tokens. Run: python scripts/test_group_avg.py"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from teacher.chunked_extract import _group_avg_tokens


def test_group_average():
    b, H, L, d, G = 1, 12, 10, 4, 2                 # rep = 6 query heads per group
    x = torch.randn(b, H, L, d)
    out = _group_avg_tokens(x, G)
    assert out.shape == (b, G, L, d), out.shape
    # group 0 == mean of the first rep=6 heads
    assert torch.allclose(out[:, 0], x[:, 0:6].mean(dim=1), atol=1e-6)
    assert torch.allclose(out[:, 1], x[:, 6:12].mean(dim=1), atol=1e-6)


if __name__ == "__main__":
    test_group_average()
    print("OK test_group_avg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/test_group_avg.py`
Expected: FAIL with `ImportError: cannot import name '_group_avg_tokens'`

- [ ] **Step 3: Write minimal implementation**

In `teacher/chunked_extract.py`:

Add module constant near `_BLOCK = 64` (line 13):

```python
_PROTO = 8              # prototypes per block (1 mean + 7 outliers), both Q and K sides
```

Add helper after `_proto_pool_seq`:

```python
def _group_avg_tokens(x, G):
    """[b,H,L,d] -> [b,G,L,d]: average the H/G query heads inside each kv-group,
    per token. Done pre-pool so each prototype is a real per-group token vector."""
    b, H, L, d = x.shape
    assert H % G == 0, f"H={H} not divisible by G={G}"
    return x.view(b, G, H // G, L, d).mean(dim=2)
```

Replace the two `_mean_pool_seq` selector-input lines in `_capture_attention` (lines 100-102). The K path uses the pre-repeat `key` (already `[1, kv=G, k_len, d]`); the Q path averages heads into groups first:

```python
    # Selector inputs as P prototypes per block (mean + outliers), so a single-token
    # spike (the needle) survives pooling. K is pre-GQA-repeat -> already per kv-group.
    # Q is averaged H->G per token before pooling so prototypes are real per-group tokens.
    q_grouped = _group_avg_tokens(query, key.shape[1])     # [1, G, q_len, d]
    pooledQ = _proto_pool_seq(q_grouped, _BLOCK, _PROTO).cpu()   # [1, G, qb, P, d]
    pooledK = _proto_pool_seq(key, _BLOCK, _PROTO).cpu()        # [1, G, kb, P, d]
    _CAPTURE_QK[module.layer_idx] = (pooledQ, pooledK)
```

Update the `_CAPTURE_QK` comment block (lines 8-12) to read `[1, G, q_blocks, P, d]` / `[1, G, k_blocks, P, d]`.

Update `get_pooled_targets`'s docstring shapes (lines 130-133) and the stacking (lines 172-173 comments) to `[1, L, G, qb, P, d]` and `[1, L, G, kb, P, d]`. The `torch.stack` lines themselves need no code change (they stack whatever shape `_CAPTURE_QK` holds).

In `scripts/extract_teacher_targets.py`, update the save dict (lines 99-113):
- change the `pooledQ`/`pooledK` shape comments to `[1, L, G, qb, P, d]` / `[1, L, G, kb, P, d]`;
- `head_dim` stays `pooledQ_stack.shape[-1]` (still d);
- `num_kv_heads` stays `pooledK_stack.shape[2]` (still G);
- add `"proto": pooledK_stack.shape[3],` after the `num_heads` line.

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/test_group_avg.py`
Expected: `OK test_group_avg`

Also confirm imports still resolve (no model run):
Run: `python -c "import teacher.chunked_extract, scripts.extract_teacher_targets" 2>&1 | tail -1`
Expected: no traceback (silent) — or a transformers import line, but no error.

- [ ] **Step 5: Commit**

```bash
git add teacher/chunked_extract.py scripts/extract_teacher_targets.py scripts/test_group_avg.py
git commit -m "Emit P-prototype Q/K selector features from extraction"
```

---

### Task 3: `targets.py` — load P axis + format guard

**Files:**
- Modify: `selector/targets.py` (`load_teacher`)
- Test: `scripts/test_targets_proto.py` (create)

**Interfaces:**
- Consumes: a `.pt` with `pooledQ [1,L,G,qb,P,d]`, `pooledK [1,L,G,kb,P,d]`, `pooled [1,L,H,qb,kb]`, and `proto` key.
- Produces: `load_teacher(path)` returns dict with `q_feat [L,G,qb,P,d]`, `k_feat [L,G,kb,P,d]`, `target [L,G,qb,kb]`, `cmask [qb,kb]`, `row_mass [L,G,qb]`, `meta` (now includes `"proto": P`). Old single-vector `.pt` (no `proto` key / 5-D pooledK) raises `KeyError` telling the user to re-extract.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_targets_proto.py`:

```python
"""Standalone test for load_teacher proto format. Run: python scripts/test_targets_proto.py"""
import os, sys, tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from selector.targets import load_teacher

L, H, G, qb, kb, P, d = 2, 4, 2, 3, 3, 8, 4


def _make_proto_pt(path):
    pooled = torch.rand(1, L, H, qb, kb)            # teacher mass (H heads)
    torch.save({
        "pooled": pooled.half(),
        "pooledQ": torch.rand(1, L, G, qb, P, d).half(),
        "pooledK": torch.rand(1, L, G, kb, P, d).half(),
        "seq_len": qb * 64, "block_size": 64, "needle_block": 1,
        "proto": P, "store_dtype": "float16",
    }, path)


def test_loads_proto_format():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.pt")
        _make_proto_pt(p)
        doc = load_teacher(p)
        assert doc["q_feat"].shape == (L, G, qb, P, d), doc["q_feat"].shape
        assert doc["k_feat"].shape == (L, G, kb, P, d), doc["k_feat"].shape
        assert doc["target"].shape == (L, G, qb, kb), doc["target"].shape
        assert doc["meta"]["proto"] == P


def test_rejects_old_format():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "old.pt")
        torch.save({                                 # old 5-D pooledK, no proto key
            "pooled": torch.rand(1, L, H, qb, kb).half(),
            "pooledQ": torch.rand(1, L, H, qb, d).half(),
            "pooledK": torch.rand(1, L, G, kb, d).half(),
            "seq_len": qb * 64, "block_size": 64, "needle_block": 1,
        }, p)
        try:
            load_teacher(p)
        except KeyError as e:
            assert "re-extract" in str(e).lower() or "re-run" in str(e).lower()
            return
        raise AssertionError("old format should have raised KeyError")


if __name__ == "__main__":
    test_loads_proto_format()
    test_rejects_old_format()
    print("OK test_targets_proto")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/test_targets_proto.py`
Expected: FAIL — `test_loads_proto_format` errors (current `load_teacher` reshapes pooledQ as 5-D / averages heads), or `test_rejects_old_format` fails because no guard exists.

- [ ] **Step 3: Write minimal implementation**

Replace the body of `load_teacher` in `selector/targets.py` (lines 26-62) with:

```python
def load_teacher(path, device="cpu", eps=1e-9):
    d = torch.load(path, map_location=device)
    if d.get("proto") is None or d["pooledK"].dim() != 6:
        raise KeyError(
            f"{path} lacks P-prototype selector features ('proto' key / 6-D pooledK). "
            f"It predates multi-prototype extraction. Re-extract with "
            f"scripts/extract_teacher_targets.py to regenerate it.")

    pooled = d["pooled"].float()      # [1, L, H, qb, kb]  teacher mass
    q_feat = d["pooledQ"].float()[0]  # [L, G, qb, P, dd]  selector input (already grouped)
    k_feat = d["pooledK"].float()[0]  # [L, G, kb, P, dd]
    pooled = pooled[0]                # [L, H, qb, kb]

    L, H, qb, kb = pooled.shape
    G = k_feat.shape[1]
    assert H % G == 0, f"H={H} not divisible by G={G}"
    rep = H // G

    # Average the teacher mass over the query heads inside each kv group -> shared G axis.
    mass = pooled.view(L, G, rep, qb, kb).mean(dim=2)         # [L, G, qb, kb]

    # Row-normalize over causal keys -> teacher marginal p^t.
    cmask = causal_block_mask(qb, kb, device=mass.device)    # [qb, kb]
    mass = mass * cmask[None, None]                          # kill any future leakage
    row = mass.sum(dim=-1, keepdim=True)                     # [L, G, qb, 1]
    target = mass / row.clamp_min(eps)

    meta = {
        "seq_len": int(d["seq_len"]), "block_size": int(d["block_size"]),
        "needle_block": int(d.get("needle_block", -1)),
        "num_layers": L, "num_groups": G, "qb": qb, "kb": kb,
        "head_dim": q_feat.shape[-1], "proto": int(d["proto"]),
    }
    return {"q_feat": q_feat, "k_feat": k_feat, "target": target,
            "cmask": cmask, "row_mass": row.squeeze(-1), "meta": meta}
```

Update the module docstring (lines 1-14) shapes: `q_feat : [L, G, qb, P, d]`, `k_feat : [L, G, kb, P, d]`, and note Q is pre-grouped in extraction (no head averaging here).

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/test_targets_proto.py`
Expected: `OK test_targets_proto`

- [ ] **Step 5: Commit**

```bash
git add selector/targets.py scripts/test_targets_proto.py
git commit -m "Load P-prototype features in targets.py; reject old format"
```

---

### Task 4: `gate.py` — MaxSim scoring with running max

**Files:**
- Modify: `selector/gate.py` (`FlatGate.forward`)
- Test: `scripts/test_gate_maxsim.py` (create)

**Interfaces:**
- Consumes: `q_feat [L,G,qb,P,d]`, `k_feat [L,G,kb,P,d]` from `targets.py`.
- Produces: `FlatGate.forward(q_feat, k_feat) -> scores [L,G,qb,kb]` = MaxSim over prototype pairs. `FlatGate.__init__(num_layers, head_dim, proj_dim=None)` unchanged.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_gate_maxsim.py`:

```python
"""Standalone test for FlatGate MaxSim. Run: python scripts/test_gate_maxsim.py"""
import os, sys, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from selector.gate import FlatGate

L, G, qb, kb, P, d = 2, 2, 3, 3, 4, 5


def test_shape():
    gate = FlatGate(L, d, proj_dim=d).eval()
    q = torch.randn(L, G, qb, P, d)
    k = torch.randn(L, G, kb, P, d)
    with torch.no_grad():
        s = gate(q, k)
    assert s.shape == (L, G, qb, kb), s.shape


def test_matches_bruteforce_maxsim():
    gate = FlatGate(L, d, proj_dim=d).eval()
    q = torch.randn(L, G, qb, P, d)
    k = torch.randn(L, G, kb, P, d)
    with torch.no_grad():
        s = gate(q, k)
        # brute force: project, all P*P pairs, max, scale
        qp = torch.einsum("lgqid,ldp->lgqip", q, gate.Wq)
        kp = torch.einsum("lgkjd,ldp->lgkjp", k, gate.Wk)
        pair = torch.einsum("lgqip,lgkjp->lgqkij", qp, kp)   # [L,G,qb,kb,P,P]
        ref = pair.amax(dim=(-1, -2)) / math.sqrt(gate.proj_dim)
    assert torch.allclose(s, ref, atol=1e-5), (s - ref).abs().max().item()


if __name__ == "__main__":
    test_shape()
    test_matches_bruteforce_maxsim()
    print("OK test_gate_maxsim")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/test_gate_maxsim.py`
Expected: FAIL — current `forward` uses 4-D einsum `lgqd,ldp` and errors on the 5-D `q_feat` (`einsum` subscript/dim mismatch).

- [ ] **Step 3: Write minimal implementation**

Replace `FlatGate.forward` in `selector/gate.py` (lines 27-31):

```python
    def forward(self, q_feat, k_feat):
        """q_feat [L,G,qb,P,d], k_feat [L,G,kb,P,d] -> raw MaxSim scores [L,G,qb,kb].
        score(q,k) = max over (query-proto i, key-proto j) of (Wq.q_i).(Wk.k_j).
        Running max over key-prototypes so [L,G,qb,kb,P,P] is never materialized."""
        qp = torch.einsum("lgqid,ldp->lgqip", q_feat, self.Wq)   # [L,G,qb,P,p]
        kp = torch.einsum("lgkjd,ldp->lgkjp", k_feat, self.Wk)   # [L,G,kb,P,p]
        best = None
        for j in range(kp.shape[3]):                             # loop key-protos
            kpj = kp[:, :, :, j, :]                              # [L,G,kb,p]
            s = torch.einsum("lgqip,lgkp->lgqik", qp, kpj)       # [L,G,qb,P,kb]
            s = s.amax(dim=3)                                    # max over query-protos
            best = s if best is None else torch.maximum(best, s)
        return best / math.sqrt(self.proj_dim)                   # [L,G,qb,kb]
```

Update the module docstring (line 3) score line to the MaxSim form.

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/test_gate_maxsim.py`
Expected: `OK test_gate_maxsim`

- [ ] **Step 5: Commit**

```bash
git add selector/gate.py scripts/test_gate_maxsim.py
git commit -m "MaxSim gate scoring over P prototypes (running max)"
```

---

## Handoff to the user (not a task)

After all four tasks are committed and green, the code is ready but produces nothing until the user re-runs the offline jobs. Report this, do not run it:

1. Re-extract 1.5B train + held-out targets: `python scripts/extract_teacher_targets.py ...` (writes 6-D proto features + `proto=8`).
2. Retrain: `python -m selector.train --targets "teacher_targets/1.5B-depth0.5/*.pt" --epochs 200 --lambda-topk 0.5 --topk 8`.
3. Eval: `python scripts/eval_gate.py --gate selector_ckpt/flat_gate.pt --targets "teacher_targets/heldout_1.5B-depth0.5/*.pt"`.

**Decisive pass signal:** training-set `ndl@8` climbs clearly off 0 (features now carry the needle). Then held-out `ndl@8` >> 0.04, with `cov@8/oracle` still ~near 1.0.
