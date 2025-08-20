"""
scoring_mahalanobis.py

Mahalanobis-based image-level anomaly scoring on MVTec-style patch embeddings.
- Streams one category at a time to keep RAM low.
- Optional PCA dimensionality reduction (trained once per backbone on a sample).
- Fits a parametric distribution on train/good patches (mean + covariance with shrinkage).
- Scores test patches with squared Mahalanobis distance; image score = mean of Top-K patch scores.

Expected cache layout (created by your feature extraction step):
    <CACHE_DIR>/<backbone>/<category>_features.pkl
    <CACHE_DIR>/<backbone>/<category>_bank.pkl

Where each *_bank.pkl contains at least:
    {"patches": np.ndarray of shape (M_train, P, D)}
and each *_features.pkl contains at least:
    {
      "patches": np.ndarray of shape (N_total, P, D),
      "meta":    pandas.DataFrame with columns ["split","path","category","raw_label","label", ...]
    }
"""

from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: FAISS PCA (keeps dependencies light, already in your env)
import faiss


# ------------------------- I/O helpers ------------------------- #
def list_categories(cache_dir: Path, backbone: str) -> List[str]:
    """
    Return sorted category names by scanning <cache_dir>/<backbone>/*_features.pkl.
    """
    d = cache_dir / backbone
    return sorted(p.stem.replace("_features", "") for p in d.glob("*_features.pkl"))


def load_category(cache_dir: Path, backbone: str, category: str) -> Tuple[Dict, Dict]:
    """
    Load per-category features and bank pickles.

    Returns:
        feats: dict with keys like {"patches": [N, P, D], "meta": DataFrame, ...}
        bank:  dict with keys like {"patches": [M, P, D], ...}
    """
    d = cache_dir / backbone
    with open(d / f"{category}_features.pkl", "rb") as f:
        feats = pickle.load(f)
    with open(d / f"{category}_bank.pkl", "rb") as f:
        bank = pickle.load(f)
    return feats, bank


# ------------------------- math helpers ------------------------ #
def apply_pca_inplace(x: np.ndarray, pca: Optional[faiss.PCAMatrix]) -> np.ndarray:
    """
    Apply FAISS PCA to a 2D array (N, D). Returns float32 contiguous array.
    If pca is None, returns x unchanged (ensured float32 contiguous).
    """
    if pca is None:
        return np.ascontiguousarray(x.astype(np.float32, copy=False))
    y = pca.apply_py(x).astype(np.float32, copy=False)
    return np.ascontiguousarray(y)


def train_pca_on_sample(
    cache_dir: Path,
    backbone: str,
    categories: List[str],
    pca_dim: Optional[int],
    sample_cap: int = 200_000,
    whitening_power: float = 0.0,  # set to -0.5 for whitening
) -> Optional[faiss.PCAMatrix]:
    """
    Train FAISS PCA on a sample of bank vectors (first category).
    """
    if pca_dim is None:
        return None
    print(f"⏳ Train PCA to {pca_dim}D on a sample...")
    sample_cat = categories[0]
    _, sample_bank = load_category(cache_dir, backbone, sample_cat)
    sb = sample_bank["patches"].reshape(-1, sample_bank["patches"].shape[-1]).astype(np.float32, copy=False)
    ntrain = min(sample_cap, sb.shape[0])
    idx = np.random.choice(sb.shape[0], size=ntrain, replace=False)
    pca = faiss.PCAMatrix(sb.shape[1], pca_dim, whitening_power)
    pca.train(sb[idx].astype(np.float32))
    assert pca.is_trained
    del sb
    gc.collect()
    print("✅ PCA trained.")
    return pca


def fit_mean_cov_shrunk(
    X: np.ndarray,
    shrinkage_alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit mean and a *shrunk* covariance matrix.

    Empirical covariance:  S = cov(X)
    Shrink towards scaled identity:  S_hat = (1 - alpha)*S + alpha*(tr(S)/D)*I

    This stabilizes inversion when samples < D or features are correlated.

    Args:
        X: [N, D] float32, rows = patch vectors (train/good only, possibly after PCA)
        shrinkage_alpha: [0..1], 0 = no shrinkage, 0.1 is a safe default

    Returns:
        mu:       [D] mean vector
        SigmaHat: [D, D] shrunk covariance matrix (float32)
    """
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=0)
    Xc = X - mu
    # ddof=1 is the unbiased estimator; we cast to float64 for stability then back to float32
    S = np.cov(Xc, rowvar=False, ddof=1).astype(np.float64, copy=False)
    D = S.shape[0]
    trace = np.trace(S)
    if trace <= 0 or not np.isfinite(trace):
        # Fallback if covariance degenerates
        trace = np.sum(np.var(X, axis=0))
    lam = trace / max(D, 1)
    S_hat = (1.0 - shrinkage_alpha) * S + shrinkage_alpha * lam * np.eye(D, dtype=np.float64)
    return mu.astype(np.float32), S_hat.astype(np.float32)


def invert_posdef(S: np.ndarray) -> np.ndarray:
    """
    Invert a (shrunk) covariance in a numerically stable way.
    Tries Cholesky, falls back to pseudo-inverse if needed.
    """
    S = S.astype(np.float64, copy=False)
    try:
        L = np.linalg.cholesky(S)
        Linv = np.linalg.solve(L, np.eye(L.shape[0], dtype=np.float64))
        S_inv = Linv.T @ Linv
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S, rcond=1e-6)
    return S_inv.astype(np.float32)


def mahalanobis_sq_batch(X: np.ndarray, mu: np.ndarray, S_inv: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
    """
    Compute squared Mahalanobis distance for rows of X in batches:
        d^2(x) = (x - mu)^T S_inv (x - mu)
    Returns array of shape [N].
    """
    X = X.astype(np.float32, copy=False)
    mu = mu.astype(np.float32, copy=False)
    S_inv = S_inv.astype(np.float32, copy=False)
    out = np.empty(X.shape[0], dtype=np.float32)
    for s in range(0, X.shape[0], batch_size):
        e = min(s + batch_size, X.shape[0])
        Xc = X[s:e] - mu
        # einsum does the per-row quadratic form efficiently
        out[s:e] = np.einsum("ij,jk,ik->i", Xc, S_inv, Xc, optimize=True)
    return out


# --------------------- scoring (streaming) --------------------- #
def score_backbone_streaming_mahalanobis(
    cache_dir: Path,
    out_dir: Path,
    backbone: str,
    k_top_patches: int = 5,
    pca_dim: Optional[int] = 128,
    pca_whitening_power: float = 0.0,  # set to -0.5 to enable whitening
    shrinkage_alpha: float = 0.1,
    query_batch_size: int = 20_000,
) -> Path:
    """
    Stream over categories for one backbone, compute image-level scores for test images
    using squared Mahalanobis distance, and append results to a single CSV.

    Image score = mean of Top-K highest patch distances.

    Args:
        cache_dir:           root of cached_dicts
        out_dir:             where to write CSV
        backbone:            e.g., "dino" or "mae"
        k_top_patches:       K for image-level top-K aggregation
        pca_dim:             None (no PCA) or output dimension (e.g., 128)
        pca_whitening_power: FAISS PCA eigen_power (0.0 = none, -0.5 ≈ whitening)
        shrinkage_alpha:     covariance shrinkage strength [0..1]
        query_batch_size:    batch size for Mahalanobis computation

    Returns:
        Path to the written CSV.
    """
    out_file = out_dir / f"scores_STREAM_{backbone}_mahalanobis_top{k_top_patches}.csv"
    wrote_header = False

    cats = list_categories(cache_dir, backbone)
    print(f"\n=== {backbone.upper()} :: STREAM Mahalanobis scoring ({len(cats)} categories) ===")

    # Train PCA once per backbone (optional)
    pca = train_pca_on_sample(
        cache_dir=cache_dir,
        backbone=backbone,
        categories=cats,
        pca_dim=pca_dim,
        sample_cap=200_000,
        whitening_power=pca_whitening_power,
    )

    for cat in cats:
        t0 = time.time()
        feats, bank = load_category(cache_dir, backbone, cat)

        # Bank: [M, P, D] -> [M*P, D]
        bank_mat = bank["patches"].reshape(-1, bank["patches"].shape[-1]).astype(np.float32, copy=False)
        bank_mat = apply_pca_inplace(bank_mat, pca)

        # Fit mean + (shrunk) covariance on train/good patches
        mu, S_hat = fit_mean_cov_shrunk(bank_mat, shrinkage_alpha=shrinkage_alpha)
        S_inv = invert_posdef(S_hat)

        # Prepare test patches
        patches = feats["patches"].astype(np.float32, copy=False)  # [N, P, D]
        meta = feats["meta"]
        is_test = meta["split"].astype(str).str.lower().eq("test").values
        test_idx = np.where(is_test)[0]
        P, D = patches.shape[1], patches.shape[2]

        X = patches[is_test].reshape(-1, D)
        X = apply_pca_inplace(X, pca)

        # Patch-level anomaly: squared Mahalanobis
        patch_scores = mahalanobis_sq_batch(X, mu, S_inv, batch_size=query_batch_size)
        patch_scores = patch_scores.reshape(-1, P)

        # Image-level: mean of Top-K highest patch scores
        tk = min(k_top_patches, P)
        order = np.argsort(patch_scores, axis=1)
        topk_idx = order[:, -tk:]
        img_scores = patch_scores[np.arange(patch_scores.shape[0])[:, None], topk_idx].mean(axis=1)

        # Rows -> CSV
        rows = []
        for j, i in enumerate(test_idx):
            rows.append(
                {
                    "idx": int(i),  # original index in the features array (may repeat across categories)
                    "path": meta.iloc[i]["path"],
                    "category": meta.iloc[i]["category"],
                    "raw_label": meta.iloc[i]["raw_label"],
                    "label": meta.iloc[i]["label"],
                    "image_score": float(img_scores[j]),
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(out_file, mode="a", header=not wrote_header, index=False)
        wrote_header = True

        # Free memory between categories
        del feats, bank, bank_mat, mu, S_hat, S_inv, patches, X, patch_scores, img_scores, df
        gc.collect()

        print(f"• {backbone}/{cat}: {len(rows)} test imgs in {time.time() - t0:.1f}s (CSV append)")

    print(f"✅ DONE → {out_file}")
    return out_file


# --------------------------- CLI hook -------------------------- #
if __name__ == "__main__":
    # Minimal CLI usage example (adjust paths to your environment)
    import argparse

    parser = argparse.ArgumentParser(description="Streamed Mahalanobis scoring.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cached_dicts root")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for CSVs")
    parser.add_argument("--backbone", type=str, required=True, help="e.g., dino or mae")
    parser.add_argument("--k_top_patches", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--pca_whitening", type=float, default=0.0, help="0.0=none, -0.5≈whitening")
    parser.add_argument("--shrinkage_alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=20000)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_backbone_streaming_mahalanobis(
        cache_dir=cache_dir,
        out_dir=out_dir,
        backbone=args.backbone,
        k_top_patches=args.k_top_patches,
        pca_dim=args.pca_dim if args.pca_dim > 0 else None,
        pca_whitening_power=args.pca_whitening,
        shrinkage_alpha=args.shrinkage_alpha,
        query_batch_size=args.batch_size,
    )
