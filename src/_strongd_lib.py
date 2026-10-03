# Auto-extracted reusable defs from ResDCGAN_GP_thermal_strongD.ipynb
import matplotlib
matplotlib.use("Agg")

# ===== cell 0 =====
import torch
import numpy as np
import os
import glob
import torch.nn as nn
import torch.autograd as autograd

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from math import log10
import pandas as pd
import dolfin  # make sure FEniCS is available

import importlib
from torchvision import transforms

HOLES = "8"

# ===== cell 1 =====
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        # you can try use activation function or not
        # return x + self.block(x)
        # x = F.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)
        return F.leaky_relu(x + self.block(x), negative_slope=0.2, inplace=False)
        # return x

# ===== cell 3 =====
class Generator(nn.Module):
    """
    Compact generator for simple 128x128 elliptical-pore microstructures.

    Key features:
    - Compact width: label_ch=128, noise_ch=128, base_ch=256
    - Two-stage noise injection:
        1) noise concatenated with label embeddings before FC
        2) noise added to the 8x8 feature map
    - Limited Conv2d refinement:
        1) after up3 at 64x64
        2) after up4 at 128x128
    - Output:
        (B, 1, 128, 128)
    """

    def __init__(
        self,
        noise_dim,
        emb_dim,            # kept for backward compatibility; not used
        img_shape,
        label_ch=128,
        noise_ch=128,
        base_ch=256,
        n_res_base=2,
        n_res_per=1
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.img_shape = img_shape
        self.size = img_shape[1] // 16

        assert img_shape == (1, 128, 128), \
            "This generator assumes 1x128x128 output."
        assert self.size == 8, \
            "This definition assumes 128x128 output."

        # -------------------------------------------------
        # Label projections
        # -------------------------------------------------
        self.label1_proj = nn.Sequential(
            nn.Linear(1, label_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(label_ch, label_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.label2_proj = nn.Sequential(
            nn.Linear(1, label_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(label_ch, label_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # -------------------------------------------------
        # First noise injection
        # noise_dim -> noise_ch
        # -------------------------------------------------
        if noise_dim == noise_ch:
            self.noise_proj = nn.Identity()
        else:
            self.noise_proj = nn.Sequential(
                nn.Linear(noise_dim, noise_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        fused_dim = label_ch * 2 + noise_ch

        # -------------------------------------------------
        # FC seed feature map
        # (B, fused_dim) -> (B, base_ch, 8, 8)
        # -------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, base_ch * self.size * self.size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # -------------------------------------------------
        # Second noise injection at 8x8 feature level
        # -------------------------------------------------
        self.noise_map_proj = nn.Sequential(
            nn.Linear(noise_dim, base_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # -------------------------------------------------
        # Optional base residual stack at 8x8
        # -------------------------------------------------
        self.base_stack = nn.Sequential(
            *[ResidualBlock(base_ch) for _ in range(n_res_base)]
        )

        # -------------------------------------------------
        # Basic upsampling block
        # -------------------------------------------------
        def up_block(cin, cout, n_res):
            layers = [
                nn.ConvTranspose2d(
                    cin,
                    cout,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            ]

            layers += [ResidualBlock(cout) for _ in range(n_res)]

            return nn.Sequential(*layers)

        # -------------------------------------------------
        # 8 -> 16 -> 32 -> 64 -> 128
        # Channels: 256 -> 128 -> 64 -> 32 -> 16
        # -------------------------------------------------
        self.up1 = up_block(base_ch,       base_ch // 2,  n_res_per)  # 256 -> 128, 8 -> 16
        self.up2 = up_block(base_ch // 2,  base_ch // 4,  n_res_per)  # 128 -> 64,  16 -> 32
        self.up3 = up_block(base_ch // 4,  base_ch // 8,  n_res_per)  # 64  -> 32,  32 -> 64
        self.up4 = up_block(base_ch // 8,  base_ch // 16, n_res_per)  # 32  -> 16,  64 -> 128

        # -------------------------------------------------
        # Conv2d refinement at 64x64
        # Helps form smoother pore structures before final resolution.
        # -------------------------------------------------
        self.refine64 = nn.Sequential(
            nn.Conv2d(
                base_ch // 8,
                base_ch // 8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(base_ch // 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # -------------------------------------------------
        # Conv2d refinement at 128x128
        # Helps clean final ellipse boundaries.
        # -------------------------------------------------
        self.refine128 = nn.Sequential(
            nn.Conv2d(
                base_ch // 16,
                base_ch // 16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(base_ch // 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # -------------------------------------------------
        # Output tail
        # -------------------------------------------------
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch // 16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight,
                a=0.2,
                nonlinearity='leaky_relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, noise, labels):
        labels = labels.float()

        l1 = labels[:, :1]
        l2 = labels[:, 1:2]

        h1 = self.label1_proj(l1)
        h2 = self.label2_proj(l2)

        # -------------------------------------------------
        # First noise injection
        # -------------------------------------------------
        z = self.noise_proj(noise)

        fused = torch.cat([h1, h2, z], dim=1)

        x = self.fc(fused)
        x = x.view(noise.size(0), -1, self.size, self.size)

        # -------------------------------------------------
        # Second noise injection at 8x8
        # -------------------------------------------------
        noise_map = self.noise_map_proj(noise)
        noise_map = noise_map.view(noise.size(0), -1, 1, 1)

        x = x + noise_map

        x = self.base_stack(x)

        # 8 -> 16
        x = self.up1(x)

        # 16 -> 32
        x = self.up2(x)

        # 32 -> 64
        x = self.up3(x)

        # Conv refinement at 64x64
        x = self.refine64(x)

        # 64 -> 128
        x = self.up4(x)

        # Conv refinement at 128x128
        x = self.refine128(x)

        img = self.tail(x)

        return img

# ===== cell 11 =====
# thermal_loader.py
import os, glob, random
from typing import Tuple, Dict, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

def _gather_ids(root: str) -> List[int]:
    ids = []
    for f in glob.glob(os.path.join(root, "thermal_*.npz")):
        try:
            ids.append(int(os.path.basename(f)[8:12]))
        except:
            continue
    ids.sort()
    return ids

def _load_image(root: str, idx: int, prefer_npy: bool = True) -> np.ndarray:
    """
    Returns HxW float32 in [0,1].
    """
    npy = os.path.join(root, f"struct_{idx:04d}.npy")
    png = os.path.join(root, f"struct_{idx:04d}.png")
    if prefer_npy and os.path.exists(npy):
        arr = np.load(npy).astype(np.float32)  # already [0,1]
    elif os.path.exists(png):
        arr = np.array(Image.open(png), dtype=np.float32) / 255.0
    else:
        raise FileNotFoundError(f"Missing image for id={idx}: {npy} or {png}")
    # ensure 128x128
    if arr.shape != (128, 128):
        arr = np.array(Image.fromarray((arr*255).astype(np.uint8)).resize((128,128), Image.NEAREST),
                       dtype=np.float32) / 255.0
    return arr

def _load_label(root: str, idx: int) -> Tuple[float, float]:
    path = os.path.join(root, f"thermal_{idx:04d}.npz")
    d = np.load(path)
    return float(d["kx"]), float(d["ky"])

def compute_label_stats(root: str, ids: List[int], mode: str = "minmax") -> Dict[str, np.ndarray]:
    """
    mode: 'minmax' or 'standard'
    Returns dict with needed stats for normalization.
    """
    kx = []
    ky = []
    for i in ids:
        x, y = _load_label(root, i)
        kx.append(x); ky.append(y)
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    if mode == "minmax":
        return {
            "mode": "minmax",
            "min": np.array([kx.min(), ky.min()], dtype=np.float64),
            "max": np.array([kx.max(), ky.max()], dtype=np.float64),
        }
    elif mode == "standard":
        return {
            "mode": "standard",
            "mean": np.array([kx.mean(), ky.mean()], dtype=np.float64),
            "std":  np.array([kx.std() + 1e-8, ky.std() + 1e-8], dtype=np.float64),
        }
    else:
        return {"mode": "none"}

def normalize_label(y: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    if stats is None or stats.get("mode", "none") == "none":
        return y
    if stats["mode"] == "minmax":
        return (y - stats["min"]) / (stats["max"] - stats["min"] + 1e-12)
    if stats["mode"] == "standard":
        return (y - stats["mean"]) / stats["std"]
    return y

def denormalize_label(y_norm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    if stats is None or stats.get("mode", "none") == "none":
        return y_norm
    if stats["mode"] == "minmax":
        return y_norm * (stats["max"] - stats["min"]) + stats["min"]
    if stats["mode"] == "standard":
        return y_norm * stats["std"] + stats["mean"]
    return y_norm

class ThermalDataset(Dataset):
    def __init__(
        self,
        root: str,
        ids: Optional[List[int]] = None,
        prefer_npy: bool = True,
        label_norm: str = "minmax",  # 'minmax', 'standard', or 'none'
        stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        self.root = root
        self.ids = _gather_ids(root) if ids is None else sorted(ids)
        # prepare stats if needed
        if stats is None and label_norm != "none":
            self.stats = compute_label_stats(root, self.ids, label_norm)
        else:
            self.stats = stats if stats is not None else {"mode": "none"}
        self.prefer_npy = prefer_npy

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img = _load_image(self.root, sample_id, self.prefer_npy)  # HxW in [0,1]
        kx, ky = _load_label(self.root, sample_id)
        y = np.array([kx, ky], dtype=np.float32)
        y = normalize_label(y, self.stats).astype(np.float32)

        # to tensors
        x = torch.from_numpy(img[None, ...])  # (1, H, W)
        y = torch.from_numpy(y)               # (2,)
        return x, y, sample_id

def make_loaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    splits=(0.9, 0.1),
    label_norm: str = "minmax",
    prefer_npy: bool = True,
):
    ids = _gather_ids(root)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(splits[0] * n)
    n_val   = int(splits[1] * n)
    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train+n_val]

    # compute stats on train only, then share
    stats = compute_label_stats(root, train_ids, label_norm)

    train_ds = ThermalDataset(root, train_ids, prefer_npy=prefer_npy, label_norm=label_norm, stats=stats)
    val_ds   = ThermalDataset(root, val_ids,   prefer_npy=prefer_npy, label_norm=label_norm, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, stats

# ===== cell 21 =====
# ==== FEniCS helpers (paste once, above your training loop) ====

_FENICS_CTX = None  # global cache for mesh/spaces

def _fenics_init(res=128):
    from dolfin import (UnitSquareMesh, FunctionSpace, MeshFunction, SubDomain,
                        near, Measure, FacetNormal, set_log_level)
    # Silence dolfin INFO logs
    try:
        from dolfin import LogLevel
        set_log_level(LogLevel.WARNING)
    except Exception:
        from dolfin import WARNING
        set_log_level(WARNING)

    mesh = UnitSquareMesh(res, res)

    # boundary tags: 1=Left, 2=Right, 3=Bottom, 4=Top
    class Left(SubDomain):
        def inside(self, x, on_b):
            return on_b and near(x[0], 0.0)
    class Right(SubDomain):
        def inside(self, x, on_b):
            return on_b and near(x[0], 1.0)
    class Bottom(SubDomain):
        def inside(self, x, on_b):
            return on_b and near(x[1], 0.0)
    class Top(SubDomain):
        def inside(self, x, on_b):
            return on_b and near(x[1], 1.0)

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    Left().mark(boundaries, 1); Right().mark(boundaries, 2)
    Bottom().mark(boundaries, 3); Top().mark(boundaries, 4)

    V  = FunctionSpace(mesh, "CG", 1)   # temperature (continuous, P1)
    V0 = FunctionSpace(mesh, "DG", 0)   # cell-wise conductivity (piecewise-constant)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    n  = FacetNormal(mesh)
    return dict(mesh=mesh, V=V, V0=V0, boundaries=boundaries, ds=ds, n=n)

def _keff_from_mask(mask01, ctx, k_matrix=1.0, k_void=1e-6, thresh=0.5, try_complement=True, eps=1e-9):
    """
    Robust keff:
      - mask01: (H,W) in [0,1] (0=black=hole, 1=white=matrix recommended)
      - <thresh -> hole (k_void), else matrix (k_matrix)
      - Computes both boundary and volumetric flux; uses the larger magnitude.
      - If both ~0, tries complement mask (swap matrix/hole).
    Returns: (keff_x, keff_y)
    """
    import numpy as np
    from dolfin import (Function, TrialFunction, TestFunction, Constant, DirichletBC,
                        solve, inner, grad, assemble, dx, cells, dot)

    mesh, V, V0 = ctx["mesh"], ctx["V"], ctx["V0"]
    boundaries, ds, n = ctx["boundaries"], ctx["ds"], ctx["n"]
    mask = np.asarray(mask01, dtype=np.float32)
    res = mask.shape[0]
    assert mask.ndim == 2 and mask.shape[0] == mask.shape[1], "mask must be square."

    def build_k(mask_img):
        k_fun = Function(V0)
        dm = V0.dofmap()
        vec = k_fun.vector().get_local()
        for cell in cells(mesh):
            dof = dm.cell_dofs(cell.index())[0]
            mp = cell.midpoint()
            ix = min(int(float(mp.x()) * res), res - 1)
            iy = min(int(float(mp.y()) * res), res - 1)
            iy_img = (res - 1) - iy  # flip image y (top->down) to FE y (bottom->up)
            is_hole = mask_img[iy_img, ix] < thresh
            vec[dof] = (k_void if is_hole else k_matrix)
        k_fun.vector().set_local(vec); k_fun.vector().apply("insert")
        return k_fun

    def solve_once(mask_img):
        k_fun = build_k(mask_img)
        T = TrialFunction(V); v = TestFunction(V)
        a = inner(k_fun * grad(T), grad(v)) * dx
        L = Constant(0.0) * v * dx

        # X direction: T(Left)=0, T(Right)=1
        Tsol = Function(V)
        solve(a == L, Tsol, [DirichletBC(V, Constant(0.0), boundaries, 1),
                             DirichletBC(V, Constant(1.0), boundaries, 2)])
        qL = assemble(-k_fun * dot(grad(Tsol), n) * ds(1))
        qx = assemble((-k_fun * grad(Tsol)[0]) * dx)
        kx = float(max(abs(qL), abs(qx)))

        # Y direction: T(Bottom)=0, T(Top)=1
        Tsol_y = Function(V)
        solve(a == L, Tsol_y, [DirichletBC(V, Constant(0.0), boundaries, 3),
                               DirichletBC(V, Constant(1.0), boundaries, 4)])
        qB = assemble(-k_fun * dot(grad(Tsol_y), n) * ds(3))
        qy = assemble((-k_fun * grad(Tsol_y)[1]) * dx)
        ky = float(max(abs(qB), abs(qy)))

        return kx, ky

    kx, ky = solve_once(mask)
    if try_complement and (kx < eps and ky < eps):
        k2x, k2y = solve_once(1.0 - mask)
        if (k2x > kx) or (k2y > ky):
            kx, ky = k2x, k2y
    return kx, ky

def to01(t):
    """Map tensor (any of [-1,1] or [0,1]) to [0,1] safely."""
    import torch
    u = t.detach().float()
    lo, hi = u.min().item(), u.max().item()
    if lo >= 0.0 and hi <= 1.0:  return u.clamp(0, 1)
    if lo >= -1.0 and hi <= 1.0: return (u.clamp(-1, 1) + 1.0) / 2.0
    return (u - lo) / (hi - lo + 1e-12)

# ===== cell 22 =====
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk

def _clean_binary_from_grayscale(img01, thresh=0.5, min_blob=12, close_iters=1):
    """
    img01: float [0,1], white=matrix.
    Returns a clean binary float mask in {0,1} with 1=matrix.
    Compatible with skimage versions that lack 'iterations' in binary_closing.
    """
    import numpy as np
    mat = (img01 >= thresh)                       # boolean
    mat = remove_small_objects(mat, min_size=min_blob)
    mat = remove_small_holes(mat, area_threshold=min_blob)

    if int(close_iters) > 0:
        se = disk(1)
        for _ in range(int(close_iters)):         # manual iterations
            mat = binary_closing(mat, footprint=se)

    return mat.astype(np.float32)                 # 1=matrix, 0=hole

