#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bathysrdt_722m_infer_v111_20261002_0525.py

BathysRDT 722M inference reference implementation.
License: BathysRDT Research License Version 1.0 (see LICENSE).
Non-commercial research and educational use only. No patent license is granted.

Features:
  - Load model.safetensors (116 tensors); head.weight is tied to embed.weight.
  - Verify the SHA-256 of the weights file against the settings file (stop on mismatch).
  - Cross-entropy / perplexity on text or on an npz file with input_ids / targets.
  - Greedy generation (reference only, no cache).
  - Self-test on CPU with a small random model.

Requirements:
  torch, safetensors, numpy. transformers (trust_remote_code) for text input.

Usage:
  python3 bathysrdt_722m_infer_v111_20261002_0525.py --self-test

  python3 bathysrdt_722m_infer_v111_20261002_0525.py \
      --weights <dir>/1400M/model.safetensors --config <dir>/1400M/config.json \
      --settings infer_settings_722m_public_v110.json --tag 1400M \
      --npz <file>.npz --rows 10

  python3 bathysrdt_722m_infer_v111_20261002_0525.py ... --text "..." --generate 32

Log tags:
  [INF.init] [INF.load] [INF.tie] [INF.cfg] [INF.eval] [INF.gen] [INF.done] [INF.err]
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "1.1.1"
SCRIPT_NAME = "bathysrdt_722m_infer"
SETTINGS_FORMAT = 2

JST = datetime.timezone(datetime.timedelta(hours=9))

HEAD_KEY = "head.weight"
EMBED_KEY = "embed.weight"


def now_jst_str():
    return datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def log(msg):
    print(msg, flush=True)


def sha256_file(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ------------------------------------------------------------------ model
class FP64LayerNorm(nn.LayerNorm):
    """LayerNorm computed in float64; input and output dtype are preserved."""

    def forward(self, x):
        orig_dtype = x.dtype
        x64 = x.double()
        out = F.layer_norm(
            x64,
            self.normalized_shape,
            self.weight.double() if self.weight is not None else None,
            self.bias.double() if self.bias is not None else None,
            self.eps,
        )
        return out.to(orig_dtype)


class RecurrentBlock(nn.Module):
    """Shared recurrent block (inference only)."""

    def __init__(self, H, nhead, ffn_dim):
        super().__init__()
        self.inject = nn.Linear(H * 2, H)
        self.norm = nn.LayerNorm(H)
        self.attn = nn.MultiheadAttention(H, nhead, batch_first=True)
        self.ff_norm = nn.LayerNorm(H)
        self.ff = nn.Sequential(
            nn.Linear(H, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, H),
        )
        self.gate = nn.Linear(H * 2, H)
        # Kept only so that the state_dict keys match the weights file. Not used at inference.
        self.raw_tau = nn.Parameter(torch.tensor(0.0))
        self.coef = 1.0

    def forward(self, s, e, mask):
        cand = self.inject(torch.cat([s, e], dim=-1))
        n = self.norm(cand)
        a, _ = self.attn(n, n, n, attn_mask=mask, is_causal=True, need_weights=False)
        cand = cand + a
        cand = cand + self.ff(self.ff_norm(cand))
        g = torch.sigmoid(self.gate(torch.cat([s, e], dim=-1)))
        g = g * self.coef
        return (1 - g) * s + g * cand


class BathysRDT722M(nn.Module):
    """BathysRDT 722M (inference only)."""

    def __init__(self, vocab, H, nhead, ffn_dim, n_pre, n_coda, state_init_scale):
        super().__init__()
        self.H = H
        self.state_init_scale = state_init_scale
        self.embed = nn.Embedding(vocab, H)

        def enc():
            return nn.TransformerEncoderLayer(
                H, nhead, ffn_dim, batch_first=True,
                activation="gelu", norm_first=True,
            )

        self.prelude = nn.ModuleList([enc() for _ in range(n_pre)])
        self.recurrent = RecurrentBlock(H, nhead, ffn_dim)
        self.coda = nn.ModuleList([enc() for _ in range(n_coda)])
        self.norm = FP64LayerNorm(H)
        self.head = nn.Linear(H, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids, depth_coef, seed):
        e = self.embed(ids)
        mask = nn.Transformer.generate_square_subsequent_mask(ids.size(1), device=ids.device)
        for layer in self.prelude:
            e = layer(e, src_mask=mask, is_causal=True)

        devices = [ids.device.index if ids.device.index is not None else torch.cuda.current_device()] \
            if ids.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            s = torch.randn_like(e) * self.state_init_scale

        for c in depth_coef:
            self.recurrent.coef = float(c)
            s = self.recurrent(s, e, mask)

        for layer in self.coda:
            s = layer(s, src_mask=mask, is_causal=True)
        return self.head(self.norm(s))


# ------------------------------------------------------------------ settings / loading
SETTINGS_REQUIRED = ("format_version", "checkpoints", "common", "tokenizer")


def validate_depth_coef(coef, n_loops):
    if not isinstance(coef, list):
        raise ValueError("depth_coef must be a list")
    if len(coef) != n_loops:
        raise ValueError(f"len(depth_coef)={len(coef)} != n_loops={n_loops}")
    for i, c in enumerate(coef):
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise ValueError(f"depth_coef[{i}] is not a number: {c!r}")
        if not math.isfinite(float(c)):
            raise ValueError(f"depth_coef[{i}] is not finite: {c!r}")
        if not (0.0 < float(c) <= 1.0):
            raise ValueError(f"depth_coef[{i}] out of range (0, 1]: {c!r}")


def load_settings(path, tag):
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    for k in SETTINGS_REQUIRED:
        if k not in st:
            raise ValueError(f"settings: missing key {k}")
    if st["format_version"] != SETTINGS_FORMAT:
        raise ValueError(f"settings: format_version={st['format_version']} (expected {SETTINGS_FORMAT})")
    if tag not in st["checkpoints"]:
        raise ValueError(f"settings: tag={tag} not found: {list(st['checkpoints'])}")
    ck = st["checkpoints"][tag]
    cm = st["common"]
    n_loops = ck.get("n_loops")
    if isinstance(n_loops, bool) or not isinstance(n_loops, int) or not (1 <= n_loops <= 64):
        raise ValueError(f"n_loops out of range: {n_loops}")
    validate_depth_coef(ck.get("depth_coef"), n_loops)
    if cm.get("dtype") not in ("bfloat16", "float32"):
        raise ValueError(f"dtype invalid: {cm.get('dtype')}")
    if isinstance(cm.get("seed"), bool) or not isinstance(cm.get("seed"), int):
        raise ValueError(f"seed must be an integer: {cm.get('seed')}")
    sis = cm.get("state_init_scale")
    if not isinstance(sis, (int, float)) or not math.isfinite(float(sis)) or float(sis) < 0:
        raise ValueError(f"state_init_scale invalid: {sis}")
    return st


def build_and_load(weights_path, config_path, settings, tag, device, skip_sha=False):
    from safetensors.torch import load_file

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    mc = cfg["model_config"]
    ck = settings["checkpoints"][tag]
    cm = settings["common"]

    expect_sha = ck.get("weights_sha256")
    if not skip_sha:
        got = sha256_file(weights_path)
        ok = (expect_sha is not None) and (got == expect_sha)
        log(f"[INF.load] weights_sha256={got} expect={expect_sha} match={ok}")
        if not ok:
            raise SystemExit("[INF.err] SHA-256 of the weights file does not match the settings. Stopping.")
    else:
        log("[INF.load] weights_sha256 check skipped (--skip-sha)")

    dtype = torch.bfloat16 if cm["dtype"] == "bfloat16" else torch.float32
    model = BathysRDT722M(
        vocab=mc["vocab"],
        H=mc["H"],
        nhead=mc["nhead"],
        ffn_dim=mc["ffn_dim"],
        n_pre=mc["n_pre"],
        n_coda=mc["n_coda"],
        state_init_scale=float(cm["state_init_scale"]),
    )
    model = model.to(dtype)

    sd = load_file(weights_path, device="cpu")
    if HEAD_KEY in sd:
        raise SystemExit("[INF.err] The weights file unexpectedly contains head.weight.")
    sd[HEAD_KEY] = sd[EMBED_KEY]
    model.load_state_dict(sd, strict=True)
    del sd
    model = model.to(device)
    model.eval()

    tied = model.head.weight is model.embed.weight
    log(f"[INF.tie] head.weight is embed.weight = {tied}")
    if not tied:
        raise SystemExit("[INF.err] Weight tying was not restored.")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[INF.load] tag={tag} params={n_params:,} dtype={dtype} device={device}")
    return model, cfg


def autocast_ctx(device, settings):
    if settings["common"].get("autocast_bf16", True) and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_forward(model, ids, settings, tag, device):
    ck = settings["checkpoints"][tag]
    cm = settings["common"]
    with torch.no_grad():
        with autocast_ctx(device, settings):
            return model(ids, depth_coef=ck["depth_coef"], seed=int(cm["seed"]))


# ------------------------------------------------------------------ metrics
def ce_per_token(logits, targets):
    v = logits.size(-1)
    return F.cross_entropy(logits.float().reshape(-1, v), targets.reshape(-1), reduction="none")


def eval_npz(model, npz_path, rows, batch, settings, tag, device):
    import numpy as np
    d = np.load(npz_path)
    ids_all = d["input_ids"][:rows]
    tgt_all = d["targets"][:rows]
    ce_sum = 0.0
    n_tok = 0
    per_row = []
    for b0 in range(0, len(ids_all), batch):
        ids = torch.as_tensor(ids_all[b0:b0 + batch], dtype=torch.long, device=device)
        tgt = torch.as_tensor(tgt_all[b0:b0 + batch], dtype=torch.long, device=device)
        lg = run_forward(model, ids, settings, tag, device)
        ce = ce_per_token(lg, tgt).reshape(ids.size(0), -1)
        for r in range(ce.size(0)):
            per_row.append(float(ce[r].mean().item()))
        ce_sum += float(ce.sum().item())
        n_tok += int(ce.numel())
        del lg
    ce_mean = ce_sum / max(n_tok, 1)
    res = {"rows": int(len(ids_all)), "tokens": n_tok, "ce": ce_mean,
           "ppl": math.exp(min(ce_mean, 50.0)), "ce_per_row": per_row}
    log(f"[INF.eval] src=npz tag={tag} rows={res['rows']} tokens={n_tok} ce={ce_mean:.6f} ppl={res['ppl']:.3f}")
    return res


def load_tokenizer(settings):
    from transformers import AutoTokenizer
    tk = settings["tokenizer"]
    tok = AutoTokenizer.from_pretrained(tk["name"], revision=tk.get("revision"), trust_remote_code=True)
    expect = tk.get("vocab_sha256")
    v = tok.get_vocab()
    blob = json.dumps(sorted(v.items(), key=lambda kv: kv[1]), ensure_ascii=False).encode("utf-8")
    got = hashlib.sha256(blob).hexdigest()
    log(f"[INF.cfg] tokenizer={tk['name']} revision={tk.get('revision')} vocab_sha256={got} match={got == expect}")
    if expect is not None and got != expect:
        raise SystemExit("[INF.err] Tokenizer vocabulary does not match the settings.")
    return tok


def encode_text(tok, text, settings, max_len, device):
    add_special = bool(settings["tokenizer"].get("add_special_tokens", False))
    ids = tok(text, add_special_tokens=add_special)["input_ids"]
    if len(ids) > max_len:
        log(f"[INF.cfg] input has {len(ids)} tokens; truncated to the first {max_len}")
        ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


def eval_text(model, tok, text, settings, tag, device):
    max_len = int(settings["common"].get("max_seq_len", 512))
    ids = encode_text(tok, text, settings, max_len, device)
    if ids.size(1) < 2:
        raise SystemExit("[INF.err] At least 2 tokens are required to compute CE.")
    lg = run_forward(model, ids, settings, tag, device)
    ce = ce_per_token(lg[:, :-1], ids[:, 1:])
    ce_mean = float(ce.mean().item())
    res = {"tokens": int(ce.numel()), "ce": ce_mean, "ppl": math.exp(min(ce_mean, 50.0))}
    log(f"[INF.eval] src=text tag={tag} tokens={res['tokens']} ce={ce_mean:.6f} ppl={res['ppl']:.3f}")
    return res


def generate_greedy(model, tok, text, n_new, settings, tag, device):
    """Greedy generation for reference. The full sequence is recomputed at every step (no cache)."""
    max_len = int(settings["common"].get("max_seq_len", 512))
    ids = encode_text(tok, text, settings, max_len, device)
    eos = tok.eos_token_id
    for _ in range(n_new):
        if ids.size(1) >= max_len:
            log(f"[INF.gen] reached max_seq_len={max_len}")
            break
        lg = run_forward(model, ids, settings, tag, device)
        nxt = int(lg[0, -1].float().argmax().item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if eos is not None and nxt == eos:
            break
    out = tok.decode(ids[0].tolist(), skip_special_tokens=False)
    log(f"[INF.gen] tag={tag} max_new_tokens={n_new} text={out!r}")
    return out


# ------------------------------------------------------------------ self-test
def self_test():
    from safetensors.torch import save_file
    fails = []

    def check(name, cond):
        print(f"[SELFTEST] {'OK ' if cond else 'NG '} {name}", flush=True)
        if not cond:
            fails.append(name)

    torch.manual_seed(123)
    mc = {"vocab": 64, "H": 16, "nhead": 2, "ffn_dim": 32, "n_pre": 1, "n_coda": 1}
    ref = BathysRDT722M(mc["vocab"], mc["H"], mc["nhead"], mc["ffn_dim"], mc["n_pre"], mc["n_coda"], 0.4)
    sd = {k: v.detach().clone().contiguous() for k, v in ref.state_dict().items() if k != HEAD_KEY}

    with tempfile.TemporaryDirectory() as td:
        wpath = os.path.join(td, "model.safetensors")
        save_file(sd, wpath, metadata={"format": "pt"})
        cpath = os.path.join(td, "config.json")
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump({"model_config": mc}, f)
        settings = {
            "format_version": SETTINGS_FORMAT,
            "checkpoints": {"T": {"weights_sha256": sha256_file(wpath), "n_loops": 3,
                                  "depth_coef": [0.7, 0.8, 0.9]}},
            "common": {"state_init_scale": 0.4, "seed": 0, "dtype": "float32",
                       "autocast_bf16": False, "max_seq_len": 32},
            "tokenizer": {"name": "dummy"},
        }
        spath = os.path.join(td, "settings.json")

        def write_settings(obj):
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(obj, f)

        write_settings(settings)
        st = load_settings(spath, "T")
        dev = torch.device("cpu")
        model, _ = build_and_load(wpath, cpath, st, "T", dev)
        check("tie restored (same object)", model.head.weight is model.embed.weight)
        check("weights loaded equal", torch.equal(model.prelude[0].linear1.weight, ref.prelude[0].linear1.weight))

        ids = torch.randint(0, 64, (2, 10))
        a = run_forward(model, ids, st, "T", dev)
        b = run_forward(model, ids, st, "T", dev)
        check("same seed -> identical logits", torch.equal(a, b))
        st2 = json.loads(json.dumps(st))
        st2["common"]["seed"] = 1
        c = run_forward(model, ids, st2, "T", dev)
        check("different seed -> different logits", not torch.equal(a, c))
        st3 = json.loads(json.dumps(st))
        st3["checkpoints"]["T"]["depth_coef"] = [0.7, 0.8, 0.95]
        c3 = run_forward(model, ids, st3, "T", dev)
        check("depth_coef changes output", not torch.equal(a, c3))
        torch.manual_seed(999)
        x1 = torch.randn(1).item()
        torch.manual_seed(999)
        _ = run_forward(model, ids, st, "T", dev)
        x2 = torch.randn(1).item()
        check("global RNG not disturbed", x1 == x2)

        causal_ids = ids.clone()
        causal_ids[:, -1] = (causal_ids[:, -1] + 1) % 64
        d = run_forward(model, causal_ids, st, "T", dev)
        check("causal: last token change keeps earlier logits", torch.allclose(a[:, :-1], d[:, :-1], atol=1e-5))

        tgt = torch.randint(0, 64, (2, 10))
        check("ce finite", bool(torch.isfinite(ce_per_token(a, tgt)).all()))

        bad_cases = {
            "coef length mismatch rejected": [0.7, 0.8],
            "coef > 1 rejected": [0.7, 0.8, 1.5],
            "coef <= 0 rejected": [0.7, 0.0, 0.9],
            "coef NaN rejected": [0.7, float("nan"), 0.9],
            "coef inf rejected": [0.7, float("inf"), 0.9],
            "coef non-number rejected": [0.7, "x", 0.9],
        }
        for name, coef in bad_cases.items():
            bad = json.loads(json.dumps(settings))
            bad["checkpoints"]["T"]["depth_coef"] = coef
            with open(spath, "w", encoding="utf-8") as f:
                f.write(json.dumps(bad, allow_nan=True))
            try:
                load_settings(spath, "T")
                check(name, False)
            except ValueError:
                check(name, True)

        bad_fmt = json.loads(json.dumps(settings))
        bad_fmt["format_version"] = 1
        write_settings(bad_fmt)
        try:
            load_settings(spath, "T")
            check("old settings format rejected", False)
        except ValueError:
            check("old settings format rejected", True)

        bad_sha = json.loads(json.dumps(settings))
        bad_sha["checkpoints"]["T"]["weights_sha256"] = "0" * 64
        write_settings(bad_sha)
        try:
            build_and_load(wpath, cpath, load_settings(spath, "T"), "T", dev)
            check("sha mismatch aborts", False)
        except SystemExit:
            check("sha mismatch aborts", True)

        sd_bad = dict(sd)
        sd_bad[HEAD_KEY] = sd[EMBED_KEY].clone()
        w2 = os.path.join(td, "model_with_head.safetensors")
        save_file(sd_bad, w2, metadata={"format": "pt"})
        write_settings(settings)
        try:
            build_and_load(w2, cpath, load_settings(spath, "T"), "T", dev, skip_sha=True)
            check("unexpected head.weight aborts", False)
        except SystemExit:
            check("unexpected head.weight aborts", True)

    print(f"[SELFTEST] {'PASS' if not fails else 'FAIL'} fails={fails}", flush=True)
    return 0 if not fails else 1


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--config")
    ap.add_argument("--settings")
    ap.add_argument("--tag")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--npz")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--text")
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--skip-sha", action="store_true")
    ap.add_argument("--out-json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())

    for k in ("weights", "config", "settings", "tag"):
        if getattr(a, k) is None:
            print(f"[INF.err] --{k} is required", flush=True)
            sys.exit(2)
    if a.out_json and os.path.exists(a.out_json):
        print(f"[INF.err] output file already exists (not overwritten): {a.out_json}", flush=True)
        sys.exit(2)

    device = torch.device(a.device)
    settings = load_settings(a.settings, a.tag)
    n_coef = len(settings["checkpoints"][a.tag]["depth_coef"])
    log(f"[INF.init] script={SCRIPT_NAME} version={__version__} start={now_jst_str()} torch={torch.__version__} "
        f"device={device} tag={a.tag} settings={os.path.basename(a.settings)} settings_md5={md5_file(a.settings)} "
        f"settings_format={SETTINGS_FORMAT} n_depth_coef={n_coef} seed={settings['common']['seed']}")
    model, cfg = build_and_load(a.weights, a.config, settings, a.tag, device, skip_sha=a.skip_sha)

    results = {"script": SCRIPT_NAME, "version": __version__, "tag": a.tag, "started": now_jst_str()}
    if a.npz:
        results["npz"] = eval_npz(model, a.npz, a.rows, a.batch, settings, a.tag, device)
    if a.text:
        tok = load_tokenizer(settings)
        results["text"] = eval_text(model, tok, a.text, settings, a.tag, device)
        if a.generate > 0:
            results["generated"] = generate_greedy(model, tok, a.text, a.generate, settings, a.tag, device)
    results["finished"] = now_jst_str()
    if a.out_json:
        with open(a.out_json, "x", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[INF.done] tag={a.tag} end={results['finished']}")


if __name__ == "__main__":
    main()
