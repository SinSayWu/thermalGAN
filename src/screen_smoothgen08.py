#!/usr/bin/env python
# ONE-SHOT screen of smoothgen08 (sigma=0.8) checkpoints: 100-sample native
# geometry (%8, npores, ellipticity, R). No smoothing, no selection.
import os, sys, json, random, re, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from geometry_metrics import hole_ellipticity_score, orientation_stats, count_holes, save_gen_png
from paths import run_dir, output_dir
# smoothgen08 uses the smoothgen Generator arch (blur buffer loaded from ckpt)
_src = open(os.path.join(os.path.dirname(__file__),
            "ResDCGAN_GP_thermal_strongD_smoothgen08.py")).read()
_m = re.search(r"# ===== notebook cell 1 =====(.*?)# ===== notebook cell 5 =====", _src, re.S)
_ns = {"torch": torch, "nn": nn, "F": F}; exec(_m.group(1), _ns)
SG = _ns["Generator"]
PORES = int(os.environ.get("PORES", "8"))         # which pore-count run to screen
D = str(run_dir(PORES))                           # THERMAL_GAN_RUNS/training_outputs_strongD_smoothgen08<PORES>
OUT = str(output_dir("oneshot_screen"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NOISE = 256; STEP = 30
tmp = f"{OUT}/_tmp{PORES}"; os.makedirs(tmp, exist_ok=True)

def gen(g, kx, ky, rng):
    z = torch.randn(1, NOISE, generator=rng, device=DEV)
    lab = torch.tensor([[kx, ky]], dtype=torch.float32, device=DEV)
    with torch.no_grad(): f = g(z, lab)
    return ((f.clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy()

eps = sorted(int(re.search(r"G_(\d+)\.pth", p).group(1)) for p in glob.glob(f"{D}/G_*.pth"))
eps = [e for e in eps if e % STEP == 0 and e >= 120]
rows = []
for ep in eps:
    g = SG(NOISE, 40, (1, 128, 128)).to(DEV)
    g.load_state_dict(torch.load(f"{D}/G_{ep}.pth", map_location=DEV, weights_only=True), strict=True)
    g.eval()
    rng = torch.Generator(device=DEV).manual_seed(1); random.seed(1)
    ns, ells, Rs = [], [], []
    for i in range(100):
        kx = round(random.uniform(0.5, 1.0), 2); ky = round(random.uniform(0.5, 1.0), 2)
        im = gen(g, kx, ky, rng); p = f"{tmp}/x.png"; save_gen_png(im, p)
        ns.append(count_holes(p, min_area=20))
        e = hole_ellipticity_score(p)
        if e > 0: ells.append(e)
        Rs.append(orientation_stats(p))
    r = dict(epoch=ep, pct8=float(np.mean(np.array(ns)==8))*100, npores=float(np.mean(ns)),
             ellipticity=float(np.mean(ells)), orientation_R=float(np.mean(Rs)))
    rows.append(r)
    print(f"sg08 G_{ep:<5d} %8={r['pct8']:5.1f}  n={r['npores']:4.2f}  ell={r['ellipticity']:.3f}  R={r['orientation_R']:.3f}", flush=True)
rows.sort(key=lambda r: (-r["pct8"], -r["ellipticity"]))
json.dump(rows, open(f"{OUT}/screen_sg08.json","w"), indent=2)
print("\n=== TOP 8 one-shot (sg08) by %8 then ellipticity ===")
for r in rows[:8]:
    print(f"G_{r['epoch']:<5d} %8={r['pct8']:5.1f}  ell={r['ellipticity']:.3f}  R={r['orientation_R']:.3f}  n={r['npores']:.2f}")
print("reference: orig G_832 %8=64 ell=0.835 | real %8=99 ell=0.843")
