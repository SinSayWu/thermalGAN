"""Geometry metrics for generated microstructures (pore count, ellipticity, orientation).

Copied verbatim from the analysis notebook used for the paper (DataAnalysis_DIVERSE.ipynb),
so the numbers match the manuscript's definitions:
  count_holes            - number of detected pores (min area filter)
  hole_ellipticity_score - mean per-pore 0.5*area-ratio + 0.5*compactness (Eq. 8)
  orientation_stats      - axial concentration R of pore orientations (1 = perfectly aligned)
  save_gen_png           - write a generator output the same way the notebook did
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from skimage import io, color, morphology, measure
from skimage.measure import regionprops


def hole_ellipticity_score(image_path: str) -> float:
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim == 3:
        img = color.rgb2gray(img)
    binary = img < 0.5
    binary = morphology.remove_small_objects(binary, 10)
    labeled = measure.label(binary)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return 0.0
    scores = []
    for region in regions:
        if region.area < 20:
            continue
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        area = region.area
        if major_axis == 0 or minor_axis == 0:
            continue
        ideal_area = np.pi * (major_axis / 2) * (minor_axis / 2)
        score = min(area, ideal_area) / max(area, ideal_area)
        perimeter = region.perimeter
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        scores.append(0.5 * score + 0.5 * compactness)
    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), 0, 1))

def orientation_stats(image_path: str, min_area: int = 30, min_axis_ratio: float = 1.10):
    """Returns axial concentration R (score) in [0,1]; 1 == perfectly aligned."""
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img.astype(float) / 255.0 if img.dtype != float else img
    binary = img_gray < 0.5
    binary = morphology.remove_small_objects(binary, 10)
    mask = (binary.astype("uint8") * 255)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    angles, weights = [], []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        (cx, cy), (MA, ma), angle_deg = cv2.fitEllipse(cnt)
        major = max(MA, ma); minor = min(MA, ma)
        if cv2.contourArea(cnt) < min_area:
            continue
        axis_ratio = major / (minor + 1e-8)
        weight = 1.0 - (minor / (major + 1e-8))
        if axis_ratio < min_axis_ratio:
            weight *= 0.05
        angles.append(angle_deg % 180.0); weights.append(weight)
    angles = np.array(angles, dtype=float); weights = np.array(weights, dtype=float)
    if len(angles) <= 1:
        return 1.0
    if weights.sum() <= 1e-12:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()
    doubled = 2.0 * np.deg2rad(angles)
    C = np.sum(weights * np.cos(doubled)); S = np.sum(weights * np.sin(doubled))
    R = np.sqrt(C * C + S * S)
    return float(np.clip(R, 0.0, 1.0))

def count_holes(image_path: str, min_area: int = 20) -> int:
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img.astype(float) / 255.0 if img.dtype != float else img
    binary = img_gray < 0.5
    binary_clean = morphology.remove_small_objects(binary, min_size=min_area)
    labeled = measure.label(binary_clean, connectivity=1)
    return len(regionprops(labeled))

def save_gen_png(im01, path):
    """Replicate generate_and_save_image: min-max normalize + gray imsave."""
    a = (im01 - im01.min()) / (im01.max() - im01.min() + 1e-8)
    plt.imsave(path, a, cmap="gray")
