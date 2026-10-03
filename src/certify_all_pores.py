#!/usr/bin/env python
# Full paper-style metrics + eval strips for the 4 smoothgen08 keeper models.
# Table-4 style: FEM median|err|+IQR over 3 anisotropy regimes (20 targets each).
# Geometry: pores==N %, ellipticity, orientation R (100 samples).
# Eval strip: Real | Generated-1 | Generated-2 for 6 val targets, FEM-validated.
import os, sys, json, random, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from _strongd_lib import make_loaders, _fenics_init, _keff_from_mask
from geometry_metrics import hole_ellipticity_score, orientation_stats, count_holes, save_gen_png
from paths import KEEPERS, keeper_path, simdata_dir, output_dir
_src=open(os.path.join(os.path.dirname(__file__),"ResDCGAN_GP_thermal_strongD_smoothgen08.py")).read()
_m=re.search(r"# ===== notebook cell 1 =====(.*?)# ===== notebook cell 5 =====",_src,re.S)
_ns={"torch":torch,"nn":nn,"F":F}; exec(_m.group(1),_ns); SG=_ns["Generator"]
OUT=str(output_dir("certify_all_pores"))          # THERMAL_GAN_OUTPUTS/certify_all_pores
FIG=str(output_dir("certify_all_pores/figures"))
DEV="cuda" if torch.cuda.is_available() else "cpu"; NOISE=256
tmp=f"{OUT}/_tmp"; os.makedirs(tmp,exist_ok=True)
INK="#15181e"; MUT="#656b76"

def load(N):
    g=SG(NOISE,40,(1,128,128)).to(DEV)
    g.load_state_dict(torch.load(keeper_path(N),
                                 map_location=DEV,weights_only=True),strict=True); g.eval(); return g
def gen(g,kx,ky,rng):
    z=torch.randn(1,NOISE,generator=rng,device=DEV); lab=torch.tensor([[kx,ky]],dtype=torch.float32,device=DEV)
    with torch.no_grad(): f=g(z,lab)
    return ((f.clamp(-1,1)+1)/2)[0,0].cpu().numpy()
def med_iqr(v):
    a=np.sort(np.abs(np.array(v))); n=len(a); return float(np.median(a)), float(a[min(int(.75*n),n-1)]-a[int(.25*n)])
def bins(seed):
    random.seed(seed); b={"x>>y":[],"x~y":[],"y>>x":[]}
    for _ in range(20):
        kx=random.randint(71,100)/100; ky=random.randint(50,int(kx*100)-20)/100; b["x>>y"].append((kx,ky))
    for _ in range(20):
        lo=random.randint(50,80); b["x~y"].append((random.randint(lo,lo+20)/100,random.randint(lo,lo+20)/100))
    for _ in range(20):
        ky=random.randint(71,100)/100; kx=random.randint(50,int(ky*100)-20)/100; b["y>>x"].append((kx,ky))
    return b

ctx=_fenics_init(res=128); results={}
for N in [1,2,4,8]:
    g=load(N)
    # --- Table 4: FEM by regime ---
    t4={}
    for reg,tg in bins(0).items():
        rng=torch.Generator(device=DEV).manual_seed(0)
        dkx,dky=[],[]
        for kx,ky in tg:
            im=gen(g,kx,ky,rng); gx,gy=_keff_from_mask(im,ctx,k_matrix=1.0,k_void=1e-6,thresh=0.5)
            dkx.append(kx-gx); dky.append(ky-gy)
        mkx,ikx=med_iqr(dkx); mky,iky=med_iqr(dky)
        t4[reg]=dict(med_kx=mkx,iqr_kx=ikx,med_ky=mky,iqr_ky=iky)
    # --- geometry 100 ---
    rng=torch.Generator(device=DEV).manual_seed(1); random.seed(1)
    ns,ells,Rs,dall_x,dall_y=[],[],[],[],[]
    for i in range(100):
        kx=round(random.uniform(0.5,1.0),2); ky=round(random.uniform(0.5,1.0),2)
        im=gen(g,kx,ky,rng); p=f"{tmp}/x.png"; save_gen_png(im,p)
        ns.append(count_holes(p,min_area=20)); e=hole_ellipticity_score(p)
        if e>0: ells.append(e)
        if N>=2: Rs.append(orientation_stats(p))
    results[N]=dict(keeper=f"{KEEPERS[N][0]}/{KEEPERS[N][1]}",
        pctN=float(np.mean(np.array(ns)==N))*100, npores=float(np.mean(ns)),
        ellipticity=float(np.mean(ells)), orientation_R=(float(np.mean(Rs)) if Rs else None),
        table4=t4)
    print(f"[{N}-pore] %{N}={results[N]['pctN']:.0f} ell={results[N]['ellipticity']:.3f} "
          f"R={results[N]['orientation_R']}", flush=True)

    # --- eval strip: Real | Gen-1 | Gen-2 for 6 val targets ---
    tl,vl,_=make_loaders(str(simdata_dir(N)),batch_size=32,num_workers=2,seed=123,
                         splits=(0.9,0.1),label_norm="none",prefer_npy=True)
    vds=vl.dataset; random.seed(7); idx=random.sample(range(len(vds)),6)
    rng=torch.Generator(device=DEV).manual_seed(5)
    fig,axes=plt.subplots(3,6,figsize=(2.0*6,2.0*3))
    for c,vi in enumerate(idx):
        real=vds[vi][0][0].numpy(); lab=vds[vi][1]; kx,ky=float(lab[0]),float(lab[1])
        axes[0,c].imshow((real>0.5),cmap="gray",vmin=0,vmax=1)
        axes[0,c].set_title(f"target ({kx:.2f},{ky:.2f})",fontsize=8,color=INK,pad=3)
        if c==0: axes[0,c].set_ylabel("Real",fontsize=9,color=INK)
        for r in (1,2):
            im=gen(g,kx,ky,rng); fx,fy=_keff_from_mask(im,ctx,k_matrix=1.0,k_void=1e-6,thresh=0.5)
            axes[r,c].imshow((im>0.5),cmap="gray",vmin=0,vmax=1)
            axes[r,c].set_xlabel(f"FEM ({fx:.2f},{fy:.2f})",fontsize=7.5,color=MUT)
            if c==0: axes[r,c].set_ylabel(f"Generated-{r}",fontsize=9,color=INK)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{N}-pore model ({KEEPERS[N][1]}): Real vs Generated (one-shot)",
                 fontsize=12,color=INK,y=1.01)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(f"{FIG}/evalstrip_p{N}.{ext}",dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"  saved evalstrip_p{N}",flush=True)

json.dump(results,open(f"{OUT}/all_pores_metrics.json","w"),indent=2)
print("saved all_pores_metrics.json")
