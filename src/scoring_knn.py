"""
scoring_knn.py

KNN image-level scoring on MVTec-style patch embeddings using FAISS.
- Streams one category at a time to keep RAM low.
- Optional PCA dimensionality reduction (trained once per backbone on a sample).
- Uses cosine (recommended) or L2 distance.
- Saves one CSV per backbone with one row per *test* image.

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

Usage (from a notebook or a small CLI):
    from pathlib import Path
    from src.scoring_knn import score_backbone_streaming

    CACHE_DIR = Path("/content/cached_dicts")     # local (fast) copy
    OUT_DIR   = Path("/content/drive/MyDrive/scores_knn")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    score_backbone_streaming(
        cache_dir=CACHE_DIR,
        out_dir=OUT_DIR,
        backbone="dino",
        metric="cosine",
        k_neighbors=5,
        topk_image=5,
        batch_size=20000,
        pca_dim=128,             # None to disable PCA
        use_fp16_gpu=True        # saves VRAM on GPU
    )
"""

from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd


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
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization: each row becomes unit length.
    """
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


# ---------------------- FAISS index builders ------------------- #
def build_flat_index(
    bank_patches: np.ndarray,
    metric: str = "cosine",
    use_fp16_gpu: bool = True,
) -> Tuple[faiss.Index, bool]:
    """
    Build a (GPU if available) flat FAISS index from bank vectors.

    Args:
        bank_patches: [N_bank, D] float32
        metric:       "cosine" (Inner Product on normalized vectors) or "l2"
        use_fp16_gpu: if True and GPU available, store vectors in FP16 on GPU to save VRAM

    Returns:
        index:    FAISS index (GPU or CPU)
        use_norm: whether cosine normalization is active
    """
    X = bank_patches.astype(np.float32, copy=False)
    d = X.shape[1]

    if metric == "cosine":
        X = l2_normalize(X)
        cpu_index = faiss.IndexFlatIP(d)  # cosine via inner product on normalized vectors
        use_norm = True
    elif metric == "l2":
        cpu_index = faiss.IndexFlatL2(d)
        use_norm = False
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")

    # Prefer GPU if available
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        if use_fp16_gpu:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
        else:
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index

    index.add(X)
    # Debug print to confirm GPU vs CPU
    print("Index type:", type(index))
    return index, use_norm


# -------------------- PCA (optional, recommended) -------------- #
def train_pca_on_sample(
    cache_dir: Path,
    backbone: str,
    categories: List[str],
    pca_dim: int,
    sample_cap: int = 200_000,
    whitening_power: float = 0.0,  # set to -0.5 for whitening
) -> Optional[faiss.PCAMatrix]:
    """
    Train a FAISS PCA matrix on a sample of bank vectors (from the first category).

    Args:
        cache_dir:        root dir of cached_dicts
        backbone:         e.g., "dino" or "mae"
        categories:       list of category names
        pca_dim:          output dimension (e.g., 128)
        sample_cap:       max number of vectors to train PCA on
        whitening_power:  eigen_power for FAISS PCA; 0.0 = no whitening, -0.5 ≈ whitening

    Returns:
        faiss.PCAMatrix or None if pca_dim is None
    """
    if pca_dim is None:
        return None

    print(f"⏳ Train PCA to {pca_dim}D on a sample...")
    # Use the first category as a representative sample source
    sample_cat = categories[0]
    _, sample_bank = load_category(cache_dir, backbone, sample_cat)

    sb = sample_bank["patches"].reshape(-1, sample_bank["patches"].shape[-1]).astype(np.float32, copy=False)
    ntrain = min(sample_cap, sb.shape[0])
    idx = np.random.choice(sb.shape[0], size=ntrain, replace=False)

    pca = faiss.PCAMatrix(sb.shape[1], pca_dim, whitening_power)  # eigen_power
    pca.train(sb[idx].astype(np.float32))
    assert pca.is_trained
    del sb
    gc.collect()
    print("✅ PCA trained.")
    return pca


def apply_pca_inplace(x: np.ndarray, pca: Optional[faiss.PCAMatrix]) -> np.ndarray:
    """
    Apply FAISS PCA to a 2D array (N, D). Returns float32 contiguous array.
    If pca is None, returns x unchanged (ensured float32 contiguous).
    """
    if pca is None:
        return np.ascontiguousarray(x.astype(np.float32, copy=False))
    y = pca.apply_py(x).astype(np.float32, copy=False)
    return np.ascontiguousarray(y)


# --------------------- scoring (streaming) --------------------- #
def score_backbone_streaming(
    cache_dir: Path,
    out_dir: Path,
    backbone: str,
    metric: str = "cosine",
    k_neighbors: int = 5,
    topk_image: int = 5,
    batch_size: int = 20_000,
    pca_dim: Optional[int] = None,
    use_fp16_gpu: bool = True,
) -> Path:
    """
    Stream over categories for one backbone, compute image-level scores for test images,
    and append results to a single CSV.

    Args:
        cache_dir:   root of cached_dicts
        out_dir:     where to write CSV
        backbone:    e.g., "dino" or "mae"
        metric:      "cosine" or "l2"
        k_neighbors: k for patch-level kNN
        topk_image:  aggregate Top-K patch scores per image
        batch_size:  query batch size for FAISS search
        pca_dim:     None (no PCA) or output dimension (e.g., 128)
        use_fp16_gpu: use FP16 storage on GPU index to save VRAM

    Returns:
        Path to the written CSV.
    """
    out_file = out_dir / f"scores_STREAM_{backbone}_{metric}_k{k_neighbors}_top{topk_image}.csv"
    wrote_header = False

    cats = list_categories(cache_dir, backbone)
    print(f"\n=== {backbone.upper()} :: STREAM KNN scoring ({len(cats)} categories) ===")

    # Train PCA once per backbone (optional)
    pca = train_pca_on_sample(
        cache_dir=cache_dir,
        backbone=backbone,
        categories=cats,
        pca_dim=pca_dim,
        sample_cap=200_000,
        whitening_power=0.0,  # set to -0.5 if you want whitening
    )

    for cat in cats:
        t0 = time.time()
        feats, bank = load_category(cache_dir, backbone, cat)

        # Bank: [M, P, D] -> [M*P, D]
        bank_mat = bank["patches"].reshape(-1, bank["patches"].shape[-1]).astype(np.float32, copy=False)
        # Optional PCA on bank
        bank_mat = apply_pca_inplace(bank_mat, pca)

        # Build index once per category
        index, use_norm = build_flat_index(bank_mat, metric=metric, use_fp16_gpu=use_fp16_gpu)

        # Features for this category
        patches = feats["patches"].astype(np.float32, copy=False)  # [N, P, D]
        meta = feats["meta"]                                      # DataFrame
        is_test = meta["split"].astype(str).str.lower().eq("test").values
        test_idx = np.where(is_test)[0]
        P, D = patches.shape[1], patches.shape[2]

        # Prepare queries: all test patches flattened to [N_test*P, D]
        X = patches[is_test].reshape(-1, D)
        X = apply_pca_inplace(X, pca)
        if use_norm:  # cosine: normalize queries too
            X = l2_normalize(X)

        # Batched search
        n = X.shape[0]
        k = k_neighbors
        all_scores = np.empty(n, dtype=np.float32)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            Dists, _ = index.search(X[s:e], k)
            # Patch score = mean k-NN distance (cosine uses 1 - sim in build_flat_index)
            # Here Dists are: cosine -> inner-products (similarities), l2 -> distances
            if use_norm:
                # For cosine we used IP on normalized vectors, so FAISS returns similarities in [0..1]
                # Convert to "distance-like" by (1 - sim), then average over k
                all_scores[s:e] = (1.0 - Dists).mean(axis=1)
            else:
                # L2: Dists are distances (or squared distances), averaging is fine for ranking
                all_scores[s:e] = Dists.mean(axis=1)

        # Reshape back to [N_test, P] patch scores
        patch_scores_test = all_scores.reshape(-1, P)

        # Image-level: mean of Top-K highest patch scores (focus on most anomalous regions)
        tk = min(topk_image, P)
        idx_sorted = np.argsort(patch_scores_test, axis=1)
        topk_idx = idx_sorted[:, -tk:]
        img_scores = patch_scores_test[np.arange(patch_scores_test.shape[0])[:, None], topk_idx].mean(axis=1)

        # Build per-image rows and append to CSV
        rows = []
        for j, i in enumerate(test_idx):
            rows.append(
                {
                    "idx": int(i),
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
        del feats, bank, bank_mat, index, patches, X, all_scores, patch_scores_test, img_scores, df
        gc.collect()
        try:
            import torch  # type: ignore
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"• {backbone}/{cat}: {len(rows)} test imgs in {time.time() - t0:.1f}s (CSV append)")

    print(f"✅ DONE → {out_file}")
    return out_file


# --------------------------- CLI hook -------------------------- #
if __name__ == "__main__":
    # Minimal CLI usage example (adjust paths to your environment)
    import argparse

    parser = argparse.ArgumentParser(description="Streamed KNN scoring with FAISS.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cached_dicts root")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for CSVs")
    parser.add_argument("--backbone", type=str, required=True, help="e.g., dino or mae")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--topk_image", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 GPU index")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_backbone_streaming(
        cache_dir=cache_dir,
        out_dir=out_dir,
        backbone=args.backbone,
        metric=args.metric,
        k_neighbors=args.k_neighbors,
        topk_image=args.topk_image,
        batch_size=args.batch_size,
        pca_dim=args.pca_dim if args.pca_dim > 0 else None,
        use_fp16_gpu=not args.no_fp16,
    )
