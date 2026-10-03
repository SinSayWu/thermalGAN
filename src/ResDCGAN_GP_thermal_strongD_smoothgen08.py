# Auto-generated from ResDCGAN_GP_thermal_strongD_smoothgen08.ipynb
import matplotlib
matplotlib.use("Agg")

# ===== notebook cell 0 =====
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

import os as _os_holes
HOLES = _os_holes.environ.get("PORES", "8")
print(f"[PORES] training {HOLES}-pore model")
from paths import simdata_dir, run_dir  # repo-relative paths; see src/paths.py

# ===== notebook cell 1 =====
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

# ===== notebook cell 3 =====
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
            # bilinear upsample + conv instead of ConvTranspose:
            # no checkerboard frequencies -> smoother pore boundaries
            layers = [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
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
        )

        # Fixed gaussian blur (7x7, sigma=0.8; reduced from 1.2 to limit pore fusion) applied to the output LOGITS
        # before tanh: the output field is band-limited by construction, so
        # its 0.5 level-set (the pore boundary) is always a SMOOTH curve.
        # The network can sharpen the transition by growing logit magnitude,
        # but it physically cannot produce pixel-scale jagged borders.
        _ax = torch.arange(7, dtype=torch.float32) - 3.0
        _g1 = torch.exp(-_ax ** 2 / (2 * 0.8 ** 2)); _g1 = _g1 / _g1.sum()
        self.register_buffer("blur_kernel", torch.outer(_g1, _g1).view(1, 1, 7, 7))

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

        logits = self.tail(x)
        logits = F.conv2d(
            F.pad(logits, (3, 3, 3, 3), mode="reflect"),
            self.blur_kernel
        )
        img = torch.tanh(logits)

        return img

# ===== notebook cell 5 =====
class Discriminator(nn.Module):
    """
    Stronger WGAN-GP critic (NO NORMALIZATION -- like the original
    haze-free critic; norm was removed because per-sample GroupNorm
    discards global contrast and lets the generator emit gray haze).

    Design goal: make the discriminator powerful enough to reject
    boundary dents (local high-frequency boundary error) and
    non-parallel pore arrangements (global orientation error), so that
    clean, mutually-aligned ellipses are enforced by the critic itself
    rather than by explicit geometric loss terms.

    Changes vs. the previous critic:
    - Wider: 64 -> 128 -> 256 -> 512 (was 32 -> 256)
    - Extra refinement conv at 8x8 to read global pore layout / alignment
    - GroupNorm (LayerNorm-equivalent, per-sample) instead of no-norm.
      BatchNorm is intentionally NOT used: it couples samples within a
      batch and breaks the per-sample gradient penalty. GroupNorm is
      per-sample and therefore GP-safe, while still stabilizing a deeper
      critic.
    - Label maps broadcast as 2 extra channels (unchanged)
    - Output: (B, 1) global critic score (unchanged)
    """

    def __init__(self, img_shape):
        super().__init__()

        assert img_shape == (1, 128, 128), \
            "This discriminator assumes 1x128x128 inputs."

        self.img_shape = img_shape
        self.img_repeats = 1

        in_ch = self.img_repeats + 2  # image + 2 label maps = 3

        def down(cin, cout, norm=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            # 128 -> 64  (no norm on the first layer, standard for WGAN-GP)
            *down(in_ch, 64, norm=False),
            # 64 -> 32
            *down(64, 128),
            # 32 -> 16
            *down(128, 256),
            # 16 -> 8
            *down(256, 512),
            # extra refinement at 8x8: lets the critic read global pore
            # arrangement / mutual orientation (parallelism)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 8x8 -> single scalar critic score
        self.head = nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0)

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

    def forward(self, img, labels):
        B = img.size(0)
        labels = labels.float()

        l1 = labels[:, :1]
        l2 = labels[:, 1:2]

        l1_map = l1.view(B, 1, 1, 1).expand(B, 1, 128, 128)
        l2_map = l2.view(B, 1, 1, 1).expand(B, 1, 128, 128)

        d_in = torch.cat([img, l1_map, l2_map], dim=1)

        x = self.features(d_in)
        validity = self.head(x)

        return validity.view(B, -1)

# ===== notebook cell 6 =====
# Gradient Penalty for WGAN-GP
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    #print(real_samples.shape)
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=real_samples.device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        # retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ===== notebook cell 7 =====
# Hyperparameters
noise_dim = 256
img_shape = (1,128,128)
batch_size = 16
num_classes = 10
emb_dim = 40
chan_num = 5

lambda_gp = 10      # was 100; canonical WGAN-GP value, lets the critic stay sharp
n_epochs = 1001
n_critic = 5        # was 1; train the critic 5x per generator update
n_generate = 1

# ===== notebook cell 11 =====
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

# ===== notebook cell 12 =====
# 1) Point to your dataset folder
root = str(simdata_dir(HOLES))  # THERMAL_GAN_DATA/simdata<PORES>

# 2) Build loaders
train_loader, val_loader, stats = make_loaders(
    root,
    batch_size=32,
    num_workers=4,
    seed=123,
    splits=(0.9, 0.1),     # tweak if you like
    label_norm="none",             # or 'standard' or 'none'
    prefer_npy=True                  # use .npy (float [0,1]) images
)

# 3) Peek at one batch
x, y, idx = next(iter(train_loader))
print(x.shape, x.dtype, x.min().item(), x.max().item())   # (B,1,128,128), float32, ~0..1
print(y.shape, y.dtype, y[:3])                            # (B,2), normalized kx,ky

# ===== notebook cell 15 =====
from torch.utils.data import DataLoader, WeightedRandomSampler

def make_weighted_sampler_region(dataset, bins, kmin, kmax, keep_epoch_len=True):
    """
    Balance sampling over [kmin, kmax]^2 using a bins×bins grid.
    Samples outside the region get weight 0.
    Returns: (sampler, n_in_region, bin_counts[bins,bins])
    """
    root, ids = dataset.root, dataset.ids

    # load labels for all ids in this split
    Y = np.array([
        [float(np.load(os.path.join(root, f"thermal_{i:04d}.npz"))["kx"]),
         float(np.load(os.path.join(root, f"thermal_{i:04d}.npz"))["ky"])]
        for i in ids
    ], dtype=np.float64)
    kx, ky = Y[:, 0], Y[:, 1]

    # region mask
    m = (kx >= kmin) & (kx <= kmax) & (ky >= kmin) & (ky <= kmax)
    if m.sum() == 0:
        raise ValueError("No samples found inside the requested region.")

    # bin edges only over [kmin, kmax]
    edges = np.linspace(kmin, kmax, bins + 1)

    # bin ids for in-region samples
    ix = np.clip(np.digitize(kx[m], edges, right=True) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(ky[m], edges, right=True) - 1, 0, bins - 1)
    bid = ix * bins + iy  # 0..bins*bins-1

    # counts per bin (avoid div/0)
    counts = np.bincount(bid, minlength=bins * bins)
    counts[counts == 0] = 1

    # inverse-frequency weights for in-region; 0 for out-of-region
    w = np.zeros(len(ids), dtype=np.float64)
    w[m] = 1.0 / counts[bid]

    weights = torch.as_tensor(w, dtype=torch.double)
    num = len(ids) if keep_epoch_len else int(m.sum())  # choose epoch length
    sampler = WeightedRandomSampler(weights, num_samples=num, replacement=True)

    return sampler, int(m.sum()), counts.reshape(bins, bins)

#--------------------------UNWEIGHTED--------------------------

from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import torch

def make_uniform_sampler_region(dataset, bins, kmin, kmax, keep_epoch_len=True):
    """
    Uniformly sample over [kmin, kmax]^2 using a bins×bins grid.
    Each bin is equally likely, regardless of how many samples it contains.
    Samples outside the region get weight 0.
    Returns: (sampler, n_in_region, bin_counts[bins,bins])
    """
    root, ids = dataset.root, dataset.ids

    Y = np.array([
        [float(np.load(os.path.join(root, f"thermal_{i:04d}.npz"))["kx"]),
         float(np.load(os.path.join(root, f"thermal_{i:04d}.npz"))["ky"])]
        for i in ids
    ], dtype=np.float64)
    kx, ky = Y[:, 0], Y[:, 1]

    m = (kx >= kmin) & (kx <= kmax) & (ky >= kmin) & (ky <= kmax)
    if m.sum() == 0:
        raise ValueError("No samples found inside the requested region.")

    edges = np.linspace(kmin, kmax, bins + 1)
    ix = np.clip(np.digitize(kx[m], edges, right=True) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(ky[m], edges, right=True) - 1, 0, bins - 1)
    bid = ix * bins + iy

    counts = np.bincount(bid, minlength=bins * bins)
    counts[counts == 0] = 1

    # uniform weights across bins
    w = np.zeros(len(ids), dtype=np.float64)
    w[m] = 1.0  # uniform probability for all in-region samples

    weights = torch.as_tensor(w, dtype=torch.double)
    num = len(ids) if keep_epoch_len else int(m.sum())
    sampler = WeightedRandomSampler(weights, num_samples=num, replacement=True)

    return sampler, int(m.sum()), counts.reshape(bins, bins)

# ===== notebook cell 16 =====
# After you already created loaders with label_norm="none"
# train_loader, val_loader, test_loader, stats = make_loaders(...)

sampler, n_in_region, bin_counts = make_weighted_sampler_region(
    train_loader.dataset, bins=13, kmin=0.55, kmax=1.0
)
print(f"In-region samples: {n_in_region}, non-empty bins: {(bin_counts>0).sum()} / {bin_counts.size}")

# Rebuild ONLY the train loader with the sampler (do NOT set shuffle)
train_loader = DataLoader(
    train_loader.dataset,
    batch_size=train_loader.batch_size,
    sampler=sampler,
    num_workers=train_loader.num_workers,
    pin_memory=True,
    drop_last=True,
)
print(f"Balanced train batches: {len(train_loader)}")

# ===== notebook cell 19 =====
# Check CUDA availability and device setup
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")
    multi_gpu = False
else:
    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} GPU(s) available.")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda:0")
    multi_gpu = num_gpus > 1
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# ===== notebook cell 20 =====
# Initialize Models, Optimizers, and Loss Function
# Initialize CUDA context
#torch.cuda.init()
generator = Generator(noise_dim, emb_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

# Optimizers — canonical WGAN-GP recipe.
# The critic now gets the same lr as G (previously it was 5x smaller),
# combined with n_critic=5, so it can actually learn to reject dents and
# non-parallel pores.
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

# ===== notebook cell 21 =====
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

# ===== notebook cell 22 =====
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

# ===== notebook cell 23 =====
# --- Output directories & prefixes ---
import os

outputDir = str(run_dir(HOLES))  # THERMAL_GAN_RUNS/training_outputs_strongD_smoothgen08<PORES>
os.makedirs(outputDir, exist_ok=True)

print(outputDir)



# checkpoint filename prefixes (used later with epoch numbers)
generatorSaveFile = os.path.join(outputDir, "G_")
discriminatorSaveFile = os.path.join(outputDir, "D_")

# make sure evaluate/save helpers can index the val set
if 'val_dataset' not in globals():
    val_dataset = val_loader.dataset

print("[INFO] outputDir:", outputDir)
print("[INFO] ckpt prefixes:", generatorSaveFile, discriminatorSaveFile)

# ===== notebook cell 24 =====
# ----------------------------
# Helpers (range-correct + robust to (img,label,id))
# ----------------------------

def save_val_strip(generator, val_dataset, device, noise_dim, output_path,
                   n_samples=10, spacing=20,
                   label_height_top=115, label_height_bottom=38,
                   font_size=20,
                   return_data=False, k_matrix=1.0, k_void=1e-6, thresh=0.5,
                   denoise=True, min_blob=12, close_iters=1):
    """
    Save a single-row strip of (Real | Generated-1 | Generated-2) triplets with labels.

    - Generator outputs are assumed in [-1,1] (tanh); mapped to [0,1] for viewing.
    - For each validation sample, two distinct seeds are used to generate two images.
    - FEniCS is computed for Generated-1 only.
    - Generated-2 is used for visual diversity comparison.

    If return_data=True, returns a list of dicts:
        {
          idx, label:(2,),
          gen01_a:(1,H,W), gen01_b:(1,H,W),
          kfenics_a:(2,) or None,
          hole_frac_a:float, hole_frac_b:float,
          seed_a:int, seed_b:int,

          # backward-compatible aliases
          gen01:(1,H,W), kfenics:(2,) or None, hole_frac:float
        }
    """

    generator.eval()

    # --- lazy FEniCS availability probe
    compute_kfenics = True
    try:
        importlib.import_module("dolfin")
    except Exception as e:
        compute_kfenics = False
        print("[FEniCS eval in strip] skipped (dolfin missing):", e)

    # --- local, version-safe denoiser
    def _clean_binary_from_grayscale(img01, thresh=0.5, min_blob=12, close_iters=1):
        """
        img01: float [0,1], white=matrix.
        Returns {0,1} float mask with 1=matrix.
        """
        try:
            from skimage.morphology import (
                remove_small_objects, remove_small_holes, binary_closing, disk
            )
            mat = (img01 >= thresh)
            mat = remove_small_objects(mat, min_size=int(min_blob))
            mat = remove_small_holes(mat, area_threshold=int(min_blob))
            if int(close_iters) > 0:
                se = disk(1)
                for _ in range(int(close_iters)):
                    mat = binary_closing(mat, footprint=se)
            return mat.astype(np.float32)
        except Exception:
            return (img01 >= thresh).astype(np.float32)

    def _postprocess_generated(gen01, run_fenics=True):
        """
        gen01: tensor (1,1,H,W) in [0,1]

        Returns:
            gen01_cpu: tensor (1,H,W)
            kfenics: (kx,ky) or None
            hole_frac: float
        """
        kfenics = None
        gen_np = gen01[0, 0].cpu().numpy()

        if denoise:
            bin_mask = _clean_binary_from_grayscale(
                gen_np, thresh=thresh, min_blob=min_blob, close_iters=close_iters
            )
            hole_frac = float((bin_mask == 0).mean())
            img_for_fenics = bin_mask
        else:
            hole_frac = float((gen_np < thresh).mean())
            img_for_fenics = gen_np.astype(np.float32)

        if compute_kfenics and run_fenics:
            try:
                kx, ky = _keff_from_mask(
                    img_for_fenics, _FENICS_CTX,
                    k_matrix=k_matrix, k_void=k_void, thresh=thresh
                )
                kfenics = (float(kx), float(ky))
            except Exception as e:
                print("[FEniCS eval in strip] single sample failed:", e)

        return gen01.squeeze(0).cpu(), kfenics, hole_frac

    # --- pick samples
    indices = torch.randperm(len(val_dataset))[:n_samples]

    gen_imgs_a, gen_imgs_b, real_imgs = [], [], []
    label_tensors = []
    idx_list = []

    kfenics_list_a, kfenics_list_b = [], []
    hole_fracs_a, hole_fracs_b = [], []
    seeds_a, seeds_b = [], []
    to_pil = transforms.ToPILImage()

    # --- lazy FEniCS context init once we know H=W
    global _FENICS_CTX
    fenics_ready = False

    for idx in indices:
        sample = val_dataset[idx]

        # handle (img,label) or (img,label,id)
        if isinstance(sample, (tuple, list)):
            real_img, label_tensor = sample[0], sample[1]
        else:
            real_img, label_tensor = sample

        lbl = label_tensor.detach().cpu()

        # two distinct seeds
        seed_a = int(torch.randint(0, 2**31 - 1, (1,)).item())
        seed_b = int(torch.randint(0, 2**31 - 1, (1,)).item())
        while seed_b == seed_a:
            seed_b = int(torch.randint(0, 2**31 - 1, (1,)).item())

        g_a = torch.Generator(device=device)
        g_a.manual_seed(seed_a)

        g_b = torch.Generator(device=device)
        g_b.manual_seed(seed_b)

        noise_a = torch.randn(1, noise_dim, generator=g_a, device=device)
        noise_b = torch.randn(1, noise_dim, generator=g_b, device=device)

        with torch.no_grad():
            gen_img_a = generator(noise_a, label_tensor.unsqueeze(0).to(device))
            gen_img_b = generator(noise_b, label_tensor.unsqueeze(0).to(device))

        gen01_a = (gen_img_a.clamp(-1, 1) + 1.0) / 2.0
        gen01_b = (gen_img_b.clamp(-1, 1) + 1.0) / 2.0

        # FEniCS init once, based on Generated-1 size
        if compute_kfenics and not fenics_ready:
            try:
                H = int(gen01_a.shape[-2])
                if "_FENICS_CTX" not in globals() or _FENICS_CTX is None:
                    _FENICS_CTX = _fenics_init(res=H)
                fenics_ready = True
            except Exception as e:
                compute_kfenics = False
                print("[FEniCS init failed in strip]:", e)

        # Compute FEniCS for Generated-1
        gen01_a_cpu, kfenics_a, hole_frac_a = _postprocess_generated(
            gen01_a, run_fenics=True
        )
        
        # Compute FEniCS for Generated-2
        gen01_b_cpu, kfenics_b, hole_frac_b = _postprocess_generated(
            gen01_b, run_fenics=True
        )

        gen_imgs_a.append(gen01_a_cpu)
        gen_imgs_b.append(gen01_b_cpu)
        real_imgs.append(real_img.cpu())
        label_tensors.append(lbl)
        idx_list.append(int(idx))

        kfenics_list_a.append(kfenics_a)
        kfenics_list_b.append(kfenics_b)
        
        hole_fracs_a.append(hole_frac_a)
        hole_fracs_b.append(hole_frac_b)
        
        seeds_a.append(seed_a)
        seeds_b.append(seed_b)
                
    # --- draw the strip
    # Use a real scalable TrueType font.
    # ImageFont.load_default() ignores font_size, so avoid it unless all TTF fonts fail.
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf",
    ]
    
    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, font_size)
            print(f"[save_val_strip] Using font: {fp}, size={font_size}")
            break
        except Exception:
            pass
    
    if font is None:
        print("[save_val_strip] WARNING: No TrueType font found. Falling back to default small font.")
        font = ImageFont.load_default()

    image_triplets = []

    for real_tensor, gen_tensor_a, gen_tensor_b, y_vec, kfk_a, kfk_b, hf_a, hf_b, sd_a, sd_b in zip(
        real_imgs, gen_imgs_a, gen_imgs_b, label_tensors,
        kfenics_list_a, kfenics_list_b, hole_fracs_a, hole_fracs_b,
        seeds_a, seeds_b
    ):
        real_pil = to_pil(real_tensor)
        gen_pil_a = to_pil(gen_tensor_a)
        gen_pil_b = to_pil(gen_tensor_b)

        w, h = real_pil.size
        triplet_w = w * 3 + spacing * 2
        triplet_h = label_height_top + h + label_height_bottom

        canvas = Image.new('RGB', (triplet_w, triplet_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        y1, y2 = float(y_vec[0]), float(y_vec[1])

        line1 = f"label ({y1:.2f}, {y2:.2f})"
        
        if (kfk_a is not None) and (kfk_b is not None):
            line2 = f"G1 k=({kfk_a[0]:.2f},{kfk_a[1]:.2f}) pore frac={hf_a:.2f}"
            line3 = f"G2 k=({kfk_b[0]:.2f},{kfk_b[1]:.2f}) pore frac={hf_b:.2f}"
        else:
            line2 = f"G1 seed={sd_a}"
            line3 = f"G2 seed={sd_b}"
        
        draw.text((5, 4),  line1, fill="white", font=font)
        draw.text((5, 28), line2, fill="white", font=font)
        draw.text((5, 52), line3, fill="white", font=font)

        # paste images
        x0 = 0
        x1 = w + spacing
        x2 = 2 * (w + spacing)

        canvas.paste(real_pil, (x0, label_height_top))
        canvas.paste(gen_pil_a, (x1, label_height_top))
        canvas.paste(gen_pil_b, (x2, label_height_top))

        # bottom captions
        draw.text((x0 + 5, label_height_top + h + 2), "Real", fill="white", font=font)
        draw.text((x1 + 5, label_height_top + h + 2), "Generated-1", fill="white", font=font)
        draw.text((x2 + 5, label_height_top + h + 2), "Generated-2", fill="white", font=font)

        image_triplets.append(canvas)

    total_w = image_triplets[0].width * n_samples + spacing * (n_samples - 1)
    total_h = image_triplets[0].height
    final_img = Image.new('RGB', (total_w, total_h), (30, 30, 30))

    for i, triplet_img in enumerate(image_triplets):
        x = i * (triplet_img.width + spacing)
        final_img.paste(triplet_img, (x, 0))

    final_img.save(output_path)

    if return_data:
        out = []
        for i in range(len(idx_list)):
            out.append({
                "idx": idx_list[i],
                "label": label_tensors[i],

                "gen01_a": gen_imgs_a[i],
                "gen01_b": gen_imgs_b[i],

                "kfenics_a": kfenics_list_a[i],
                "kfenics_b": kfenics_list_b[i],
                
                "hole_frac_a": hole_fracs_a[i],
                "hole_frac_b": hole_fracs_b[i],

                "seed_a": seeds_a[i],
                "seed_b": seeds_b[i],

                # backward-compatible aliases = first generated image
                "gen01": gen_imgs_a[i],
                "kfenics": kfenics_list_a[i],
                "hole_frac": hole_fracs_a[i],
            })
        return out


@torch.no_grad()
def evaluate_and_visualize(generator, val_loader, val_dataset, device, noise_dim, out_path, n_samples=10):
    # Save strip and grab the same generated samples + labels
    samples = save_val_strip(
        generator, val_dataset, device, noise_dim, out_path,
        n_samples=n_samples,
        font_size=20,
        label_height_top=115,
        label_height_bottom=38,
        return_data=True
    )

    # ---- regular image metrics on the whole val loader ----
    generator.eval()
    mse_sum = psnr_sum = ssim_sum = 0.0
    count = 0

    for batch in val_loader:
        if isinstance(batch, (tuple, list)):
            img_real, label = batch[0], batch[1]
        else:
            img_real, label = batch

        img_real = img_real.to(device)
        B = img_real.size(0)

        noise = torch.randn(B, noise_dim, device=device)
        img_fake = generator(noise, label.to(device))
        img_fake01 = (img_fake.clamp(-1, 1) + 1.0) / 2.0

        mse_b = F.mse_loss(
            img_fake01, img_real, reduction="none"
        ).view(B, -1).mean(dim=1)

        mse_sum += mse_b.sum().item()
        psnr_sum += (
            10.0 * torch.log10(1.0 / mse_b.clamp_min(1e-12))
        ).sum().item()

        rc = img_real.squeeze(1).detach().cpu().numpy()
        fc = img_fake01.squeeze(1).detach().cpu().numpy()

        for i in range(B):
            ssim_sum += ssim(rc[i], fc[i], data_range=1.0)

        count += B

    mse = mse_sum / count
    psnr = psnr_sum / count
    ssim_val = ssim_sum / count

    return mse, psnr, ssim_val, samples

# ===== notebook cell 25 =====
# ----------------------------
# Training loop (WGAN-GP; strong critic enforces geometry)
# ----------------------------
if multi_gpu:
    print("Enabling DataParallel for multi-GPU training.")
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

generator.to(device)
discriminator.to(device)

# make sure this exists
val_dataset = val_loader.dataset

epoch_list, mse_list, psnr_list, ssim_list = [], [], [], []
kx_err = []
ky_err = []

# ----------------------------
# Train the critic 5x per generator step (canonical WGAN-GP).
# Geometry (clean ellipses + parallel alignment) is enforced by the
# critic, NOT by explicit shape losses:
#   - pixel-level diversity REMOVED (it rewarded boundary dents)
#   - TV smoothness loss REMOVED (explicit shape prior)
# Kept:
#   - coarse-arrangement diversity only, to keep the non-unique inverse
#     map from collapsing (diversifies pore PLACEMENT, not edge shape)
#   - a light binary push so the [-1,1] output is near-binary before the
#     0.5 threshold (output FORMAT, not a geometric constraint)
# ----------------------------
n_critic = 5
lambda_div = 1.0     # weight on coarse-arrangement diversity
lambda_bin = 0.05    # weight on binary-output push
lambda_tv = 0.10     # TV smoothness (anti-dent), always on

# --- CURRICULUM: shape first, then haze ---
# Phase 1 (epoch < shape_epochs): NO hard binarization -> let the
#   no-norm critic + coarse diversity + TV grow clean large parallel
#   ellipses (pixel-diversity is intentionally ABSENT: it fragments).
# Phase 2 (epoch >= shape_epochs): ramp a hard BCE-to-hard-threshold
#   loss 0->lambda_hardbin_max over hardbin_ramp epochs, forcing every
#   pixel to 0/1 WITHOUT rewarding fragmentation. Critic+TV hold shape.
shape_epochs = 150
hardbin_ramp = 50
lambda_hardbin_max = 1.0

for epoch in range(n_epochs):
    generator.train()
    discriminator.train()

    for batch_idx, batch in enumerate(train_loader):

        # accept (img,label,...) or (img,label)
        if isinstance(batch, (tuple, list)):
            imgs, labels = batch[0], batch[1]
        else:
            imgs, labels = batch

        real_imgs = imgs.to(device)       # [B,1,H,W] in [0,1]
        labels = labels.to(device)        # [B,2]
        B = real_imgs.size(0)

        # match generator tanh output range
        real_in = real_imgs * 2.0 - 1.0   # [0,1] -> [-1,1]

        # ============================================================
        # Train Discriminator / Critic every batch
        # ============================================================
        optimizer_D.zero_grad()

        with torch.no_grad():
            z_D = torch.randn(B, noise_dim, device=device)
            fake_in = generator(z_D, labels)     # [-1,1]

        real_validity = discriminator(real_in, labels)
        fake_validity_D = discriminator(fake_in, labels)

        gradient_penalty = compute_gradient_penalty(
            discriminator,
            real_in,
            fake_in,
            labels
        )

        d_loss = (
            -real_validity.mean()
            + fake_validity_D.mean()
            + lambda_gp * gradient_penalty
        )

        d_loss.backward()
        optimizer_D.step()

        # ============================================================
        # Train Generator every n_critic batches
        # ============================================================
        if batch_idx % n_critic == 0:
            optimizer_G.zero_grad()

            z1 = torch.randn(B, noise_dim, device=device)
            z2 = torch.randn(B, noise_dim, device=device)

            fake_imgs_1 = generator(z1, labels)   # [-1,1]
            fake_imgs_2 = generator(z2, labels)   # [-1,1]

            fake_validity_G = discriminator(fake_imgs_1, labels)

            # ----------------------------
            # 1. WGAN generator loss (the critic does the heavy lifting)
            # ----------------------------
            g_adv_loss = -fake_validity_G.mean()

            fake01_1 = (fake_imgs_1 + 1.0) / 2.0
            fake01_2 = (fake_imgs_2 + 1.0) / 2.0

            # ----------------------------
            # 2. Coarse-ARRANGEMENT diversity ONLY
            #    (no pixel/boundary diversity -> does not reward dents)
            #    Operates on a 16x16 pooled occupancy map, so it rewards
            #    different pore PLACEMENTS for different noise, keeping the
            #    non-unique inverse map from collapsing.
            # ----------------------------
            sharp1 = torch.sigmoid(10.0 * (fake01_1 - 0.5))
            sharp2 = torch.sigmoid(10.0 * (fake01_2 - 0.5))

            low1 = F.avg_pool2d(sharp1, kernel_size=16, stride=16)  # (B,1,8,8)
            low2 = F.avg_pool2d(sharp2, kernel_size=16, stride=16)

            spatial_diff = torch.mean(torch.abs(low1 - low2), dim=[1, 2, 3])
            z_diff = torch.mean(torch.abs(z1 - z2), dim=1)

            diversity_score = torch.mean(spatial_diff / (z_diff + 1e-8))
            div_term = lambda_div * diversity_score

            # ----------------------------
            # 3. Light binary-output push (FORMAT only, not a shape prior)
            # ----------------------------
            binary_loss = torch.mean(fake01_1 * (1.0 - fake01_1))

            # ----------------------------
            # 3b. TV smoothness (anti-dent) -- always on, both phases
            # ----------------------------
            # ISOTROPIC gradient smoothness (anisotropic TV favors staircase
            # edges; the isotropic form penalizes boundary length regardless of
            # direction -> round, smooth pore boundaries)
            _dx = fake01_1[:, :, 1:, 1:] - fake01_1[:, :, :-1, 1:]
            _dy = fake01_1[:, :, 1:, 1:] - fake01_1[:, :, 1:, :-1]
            tv_loss = torch.mean(torch.sqrt(_dx ** 2 + _dy ** 2 + 1e-8))

            # ----------------------------
            # 3c. PHASE-2 HARD binarization: OFF until shape_epochs, then ramp.
            #     BCE against the output's OWN detached hard threshold repels
            #     each pixel away from 0.5 toward 0/1 (strong gradient at gray),
            #     crisping without fragmenting.
            # ----------------------------
            hard_target = (fake01_1 > 0.5).float().detach()
            _eps = 1e-6
            p_clamped = fake01_1.clamp(_eps, 1.0 - _eps)
            hardbin_loss = F.binary_cross_entropy(p_clamped, hard_target)
            if epoch < shape_epochs:
                lambda_hardbin = 0.0
            else:
                lambda_hardbin = min((epoch - shape_epochs) / float(hardbin_ramp), 1.0) * lambda_hardbin_max

            # ----------------------------
            # 4. Final generator loss
            # ----------------------------
            g_loss = (
                g_adv_loss
                - div_term
                + lambda_bin * binary_loss
                + lambda_tv * tv_loss
                + lambda_hardbin * hardbin_loss
            )

            g_loss.backward()
            optimizer_G.step()

        # ============================================================
        # Logging
        # ============================================================
        if batch_idx % 200 == 0:
            if batch_idx % n_critic == 0:
                print(
                    f"[Epoch {epoch}/{n_epochs}] "
                    f"[Batch {batch_idx}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] "
                    f"[G loss: {g_loss.item():.4f}] "
                    f"[G adv: {g_adv_loss.item():.4f}] "
                    f"[Div: {diversity_score.item():.4f}] "
                    f"[Div term: {div_term.item():.4f}] "
                    f"[SpatialDiv: {spatial_diff.mean().item():.4f}] "
                    f"[Bin: {binary_loss.item():.4f}] "
                    f"[TV: {tv_loss.item():.4f}] "
                    f"[HardBin: {hardbin_loss.item():.4f} (lhb {lambda_hardbin:.2f})] "
                    f"[GP: {gradient_penalty.item():.4f}] "
                    f"[real: {real_validity.mean().item():.4f}] "
                    f"[fake_D: {fake_validity_D.mean().item():.4f}] "
                    f"[fake_G: {fake_validity_G.mean().item():.4f}]"
                )
            else:
                print(
                    f"[Epoch {epoch}/{n_epochs}] "
                    f"[Batch {batch_idx}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] "
                    f"[G skipped] "
                    f"[GP: {gradient_penalty.item():.4f}] "
                    f"[real: {real_validity.mean().item():.4f}] "
                    f"[fake_D: {fake_validity_D.mean().item():.4f}]"
                )

    # ---------------------
    # Eval + image strip every 5 epochs
    # ---------------------
    if epoch % 10 == 0:
        out_path = os.path.join(outputDir, f"epoch{epoch}_val_comparison.png")
        mse_epoch, psnr_epoch, ssim_epoch, samples = evaluate_and_visualize(
            generator, val_loader, val_dataset, device, noise_dim, out_path, n_samples=10
        )
        epoch_list.append(epoch)
        mse_list.append(mse_epoch)
        psnr_list.append(psnr_epoch)
        ssim_list.append(ssim_epoch)
        print(f"[Eval @ epoch {epoch}] MSE: {mse_epoch:.6f} | PSNR: {psnr_epoch:.3f} | SSIM: {ssim_epoch:.4f}")
    
        # ---- FEniCS cross-check USING THE SAME IMAGES FROM THE STRIP ----
        kx_epoch = float('nan')
        ky_epoch = float('nan')
        try:
            # print("[FEniCS eval] idx | label(kx,ky) -> fenics(kx,ky) | abs err  [range, holes]")
    
            kx_sum = 0.0
            ky_sum = 0.0
            sample_count = 0
    
            for s in samples:
                idx = int(s["idx"])
                y = s["label"]
                y_np = y.detach().cpu().numpy().astype(float)
                kfk = s["kfenics"]                 # already computed inside save_val_strip
                gen01_t = s["gen01"]               # (1,H,W) tensor in [0,1]
                gen01 = gen01_t.squeeze(0).cpu().numpy()
                mn, mx = float(gen01.min()), float(gen01.max())
                holes = float(s["hole_frac"])
    
                if kfk is None:
                    # (Only if FEniCS was unavailable in save_val_strip)
                    kx, ky = _keff_from_mask(gen01, _FENICS_CTX, k_matrix=1.0, k_void=1e-6, thresh=0.5)
                else:
                    kx, ky = kfk
    
                ex = abs(kx - float(y_np[0]))
                ey = abs(ky - float(y_np[1]))
                kx_sum += ex
                ky_sum += ey
                sample_count += 1
    
                # print(f"{idx:5d} | ({y_np[0]:.3f},{y_np[1]:.3f}) -> ({kx:.3f},{ky:.3f}) | "
                #       f"(|{ex:.3f}|, |{ey:.3f}|)  [range {mn:.3f}..{mx:.3f}, holes={holes:.3f}]")
    
            if sample_count > 0:
                kx_epoch = kx_sum / sample_count
                ky_epoch = ky_sum / sample_count
    
            print(f"average kx error: {kx_epoch:.4f}   average ky error: {ky_epoch:.4f}")
    
        except Exception as e:
            print("[FEniCS eval] skipped:", e)
    
        try:
            hz = []
            for s in samples:
                im = s["gen01_a"].squeeze(0).cpu().numpy()
                hz.append(float(((im > 0.2) & (im < 0.8)).mean()))
            print(f"[HAZE] mean ambiguous-frac: {np.mean(hz):.4f}  (orig~0.005, hazy~0.045)")
        except Exception as e:
            print("[HAZE] skipped:", e)

        try:
            from skimage.measure import label as _sklabel, regionprops as _skprops
            asps = []; Rs = []; ns = []
            for s in samples:
                im = s["gen01_a"].squeeze(0).cpu().numpy()
                bm = (im > 0.5).astype(np.uint8)        # 1=matrix, 0=hole
                lab = _sklabel(bm == 0)
                props = [p for p in _skprops(lab) if p.area >= 15]
                if len(props) < 2:
                    continue
                aspects = []; angs = []; ws = []
                for p in props:
                    mn = max(p.minor_axis_length, 1e-6); mj = p.major_axis_length
                    aspects.append(mj / mn)
                    angs.append(p.orientation)
                    ws.append(1.0 - mn / max(mj, 1e-6))
                angs = np.array(angs); ws = np.array(ws)
                C = np.sum(ws * np.cos(2 * angs)); S = np.sum(ws * np.sin(2 * angs))
                R = np.sqrt(C * C + S * S) / max(ws.sum(), 1e-6)
                asps.append(float(np.mean(aspects))); Rs.append(float(R)); ns.append(len(props))
            if asps:
                print(f"[SHAPE] aspect(elong): {np.mean(asps):.2f}  align R(parallel): {np.mean(Rs):.3f}  npores/img: {np.mean(ns):.1f}  (want asp>=2.8, R>=0.96, npores~8)")
        except Exception as e:
            print("[SHAPE] skipped:", e)

        try:
            _g = generator.module if isinstance(generator, torch.nn.DataParallel) else generator
            _g.eval()
            with torch.no_grad():
                _lr = 0.5 + 0.5 * torch.rand(20, 2, device=device)
                _zr = torch.randn(20, noise_dim, device=device)
                _fr = ((_g(_zr, _lr).clamp(-1, 1) + 1) / 2)[:, 0].cpu().numpy()
            _hr = float(np.mean(((_fr > 0.2) & (_fr < 0.8)).reshape(20, -1).mean(1)))
            print(f"[HAZE-RAND] {_hr:.4f}  (random uniform labels)")
            with torch.no_grad():
                _l0 = (0.5 + 0.5 * torch.rand(1, 2, device=device)).repeat(6, 1)
                _z6 = torch.randn(6, noise_dim, device=device)
                _f6 = (((_g(_z6, _l0).clamp(-1, 1) + 1) / 2) > 0.5).float()[:, 0].reshape(6, -1)
            _pd = [float((_f6[a] != _f6[b]).float().mean()) for a in range(6) for b in range(a + 1, 6)]
            print(f"[DIV] within-label diversity: {np.mean(_pd):.4f}  (healthy~0.17, collapsed~0.005)")
            from skimage import measure as _skm, morphology as _skmo
            _es = []; _n8 = 0; _nt = 0
            for s in samples:
                _im = s["gen01_a"].squeeze(0).cpu().numpy()
                _b = _skmo.remove_small_objects(_im < 0.5, 20)
                _regs = [r for r in _skm.regionprops(_skm.label(_b, connectivity=1)) if r.area >= 20]
                _nt += 1; _n8 += (len(_regs) == 8)
                _sc = []
                for r in _regs:
                    if r.major_axis_length <= 0 or r.minor_axis_length <= 0:
                        continue
                    _ia = np.pi * (r.major_axis_length / 2) * (r.minor_axis_length / 2)
                    _sc.append(0.5 * min(r.area, _ia) / max(r.area, _ia)
                               + 0.5 * (4 * np.pi * r.area) / (r.perimeter ** 2 + 1e-8))
                if _sc:
                    _es.append(np.mean(_sc))
            print(f"[PAPER] ellipticity={np.mean(_es):.3f} (real 0.843, orig 0.835)  "
                  f"pores==8: {_n8}/{_nt}")
            generator.train()
        except Exception as e:
            print("[PAPER-METRICS] skipped:", e)

        kx_err.append(kx_epoch)
        ky_err.append(ky_epoch)
    
        df = pd.DataFrame({
            "Epoch": epoch_list,
            "MSE": mse_list,
            "PSNR": psnr_list,
            "SSIM": ssim_list,
            "kx_err": kx_err,
            "ky_err": ky_err
        })
        df.to_csv(os.path.join(outputDir, "metrics_log.csv"), index=False)
        print("Metrics saved to metrics_log.csv")


    # ---------------------
    # Save checkpoints
    # ---------------------
    if epoch % 10 == 0 and epoch >= 20:
        g_state = generator.module.state_dict() if isinstance(generator, torch.nn.DataParallel) else generator.state_dict()
        d_state = discriminator.module.state_dict() if isinstance(discriminator, torch.nn.DataParallel) else discriminator.state_dict()
        torch.save(g_state, f"{generatorSaveFile}{epoch}.pth")
        torch.save(d_state, f"{discriminatorSaveFile}{epoch}.pth")
        print(f"[ckpt] saved G_{epoch}.pth and D_{epoch}.pth -> {outputDir}")

