#!/usr/bin/env python
# Within-label diversity score for each GAN model (smoothgen / nobin / original) x
# pore count. Exact campaign metric (ablation.py): generate K samples at the SAME
# label with K different noise seeds, threshold at 0.5, return the mean pairwise
# Hamming distance (fraction of differing pixels) over all K-choose-2 pairs.
# Averaged over several labels for a stable estimate. High = diverse; ~0.01 = collapsed.
import os, sys, json, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from _orig_big_gen import Generator as BigG

from paths import KEEPERS, keeper_path, output_dir
OUT = str(output_dir("eval_diversity"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NOISE = 256
K = 8                                    # samples per label (28 pairs), matches campaign
LABELS = [(0.75, 0.60), (0.60, 0.80), (0.90, 0.55), (0.70, 0.70)]  # avg over a few targets
SEED = 3

_sg = open(os.path.join(os.path.dirname(__file__), "ResDCGAN_GP_thermal_strongD_smoothgen08.py")).read()
_m = re.search(r"# ===== notebook cell 1 =====(.*?)# ===== notebook cell 5 =====", _sg, re.S)
_ns = {"torch": torch, "nn": nn, "F": F}; exec(_m.group(1), _ns); SG = _ns["Generator"]

SMOOTHGEN = {N: str(keeper_path(N)) for N in KEEPERS}   # released models (weights/)

# Optional ablation comparisons (Table VIII). Their weights are not in this repo; to include
# them, point these variables at a JSON file mapping pore count -> checkpoint path, e.g.
#   {"1": "/path/G_660.pth", "2": "...", "4": "...", "8": "..."}
def _optional(var):
    p = os.environ.get(var)
    return {int(k): v for k, v in json.load(open(p)).items()} if p else {}
NOBIN = _optional("THERMAL_GAN_NOBIN_CKPTS")        # same architecture, no inverse-bin sampling
ORIGINAL = _optional("THERMAL_GAN_LARGE_CKPTS")     # large-capacity generator (src/_orig_big_gen.py)
FAMILIES = [(f, r) for f, r in [("smoothgen", SMOOTHGEN), ("nobin", NOBIN), ("original", ORIGINAL)] if r]


def load_sg(p):
    g = SG(NOISE, 40, (1, 128, 128)).to(DEV)
    g.load_state_dict(torch.load(p, map_location=DEV, weights_only=True), strict=True); g.eval(); return g

def load_orig(p):
    g = BigG(NOISE, 0, (1, 128, 128), label_ch=512, noise_ch=512, base_ch=1024,
             n_res_base=3, n_res_per=4).to(DEV)
    g.load_state_dict(torch.load(p, map_location=DEV, weights_only=True), strict=True); g.eval(); return g


def diversity(g):
    scores = []
    for (kx, ky) in LABELS:
        rng = torch.Generator(device=DEV).manual_seed(SEED)
        lab = torch.tensor([[kx, ky]] * K, dtype=torch.float32, device=DEV)
        z = torch.randn(K, NOISE, generator=rng, device=DEV)
        with torch.no_grad(): f = g(z, lab)
        b = (((f.clamp(-1, 1) + 1) / 2) > 0.5).float()[:, 0].reshape(K, -1)
        scores.append(np.mean([float((b[a] != b[c]).float().mean())
                               for a in range(K) for c in range(a + 1, K)]))
    return float(np.mean(scores))


results = {}
for fam, reg in FAMILIES:
    loader = load_orig if fam == "original" else load_sg
    for N in sorted(reg):
        g = loader(reg[N]); d = diversity(g)
        results[f"{fam}_{N}"] = round(d, 4)
        print(f"[{fam}_{N}] diversity = {d:.4f}", flush=True)
        del g
        if DEV == "cuda": torch.cuda.empty_cache()

meta = dict(description="Within-label diversity = mean pairwise Hamming distance over K one-shot "
            "samples at the same label (avg over several labels). ~0.01=mode-collapsed, higher=diverse.",
            K=K, labels=LABELS, seed=SEED, models=results)
json.dump(meta, open(f"{OUT}/eval_diversity.json", "w"), indent=2)

FAMS = [f for f, _ in FAMILIES]; PORES = [1, 2, 4, 8]
L = ["Within-label diversity score (mean pairwise Hamming; ~0.01=collapsed, higher=diverse)", "",
     f"{'model':>10s} | " + " | ".join(f"{N:^7d}" for N in PORES) + "   (# pores)", "-" * 44]
for f in FAMS:
    L.append(f"{f:>10s} | " + " | ".join(f"{results[f'{f}_{N}']:^7.3f}" for N in PORES))
txt = "\n".join(L)
open(f"{OUT}/eval_diversity.txt", "w").write(txt + "\n")
print("\n" + txt)
print(f"\nsaved {OUT}/eval_diversity.{{json,txt}}")
