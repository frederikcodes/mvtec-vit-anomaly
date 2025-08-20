import math, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor

# ========== DEVICE ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== TRANSFORMS (NO center crop) ==========
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])

# ========== DATASET ==========
class ImagePathDataset(Dataset):
    def __init__(self, img_paths, labels, categories, splits, transform=None):
        self.imgs = img_paths
        self.labels = labels
        self.categories = categories
        self.splits = splits
        self.transform = transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        p = self.imgs[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        meta = {
            "path": str(p),
            "category": self.categories[idx],
            "raw_label": self.labels[idx],
            "label": "good" if self.labels[idx] == "good" else "defect",
            "split": self.splits[idx],
        }
        return img, meta

def collate_keep_meta(batch):
    imgs, metas = zip(*batch)
    return torch.stack(imgs), list(metas)

def collect_images(cat_dir: Path, cat: str):
    paths, labels, categories, splits = [], [], [], []
    for split in ["train", "test"]:
        split_dir = cat_dir / split
        if not split_dir.exists(): continue
        for defect_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            lbl = defect_dir.name
            for img_path in defect_dir.rglob("*.png"):
                paths.append(img_path)
                labels.append(lbl)
                categories.append(cat)
                splits.append(split)
    return paths, labels, categories, splits

# ========== DINOv2 EXTRACTOR ==========
class DinoFeatureExtractor:
    def __init__(self, model_name="dinov2_vitb14", project_dir=Path("/content/drive/MyDrive")):
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(DEVICE).eval()
        self.project_dir = Path(project_dir)
        self.feat_dir = self.project_dir / "features_dinov2_b14"
        self.bank_dir = self.project_dir / "featurebanks" / "dinov2_b14"
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        self.bank_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        f = self.model(x)
        return F.normalize(f, dim=1)

    def run_all(self, cat: str, data_dir: Path, batch_size=128, overwrite=False):
        out_npz = self.feat_dir / f"{cat}_dinov2_vitb14.npz"
        out_csv = self.feat_dir / f"{cat}_meta.csv"
        if not overwrite and out_npz.exists(): return

        paths, labels, cats, splits = collect_images(data_dir / cat, cat)
        ds = ImagePathDataset(paths, labels, cats, splits, preprocess)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_keep_meta)
        feats, metas = [], []

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            for imgs, metas_batch in tqdm(dl, desc=f"[DINOv2 ALL] {cat}"):
                imgs = imgs.to(DEVICE)
                f = self.extract(imgs).cpu().numpy()
                feats.append(f)
                metas.extend(metas_batch)

        np.savez_compressed(out_npz, features=np.concatenate(feats))
        pd.DataFrame(metas).to_csv(out_csv, index=False)

    def run_bank(self, cat: str, data_dir: Path, batch_size=128, overwrite=False):
        out_npz = self.bank_dir / f"{cat}_bank_dinov2_b14.npz"
        out_csv = self.bank_dir / f"{cat}_bank_meta.csv"
        if not overwrite and out_npz.exists(): return

        good_dir = data_dir / cat / "train" / "good"
        paths = sorted(good_dir.rglob("*.png"))
        labels = ["good"] * len(paths)
        cats = [cat] * len(paths)
        splits = ["train"] * len(paths)
        ds = ImagePathDataset(paths, labels, cats, splits, preprocess)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_keep_meta)
        feats, metas = [], []

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            for imgs, metas_batch in tqdm(dl, desc=f"[DINOv2 BANK] {cat}"):
                imgs = imgs.to(DEVICE)
                f = self.extract(imgs).cpu().numpy()
                feats.append(f)
                metas.extend(metas_batch)

        np.savez_compressed(out_npz, features=np.concatenate(feats))
        pd.DataFrame(metas).to_csv(out_csv, index=False)

# ========== MAE EXTRACTOR ==========
# ========== MAE EXTRACTOR (masking disabled, 256x256 via processor) ==========
class MAEFeatureExtractor:
    def __init__(self, model_name="facebook/vit-mae-base", project_dir=Path("/content/drive/MyDrive")):
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.feat_dim = 768
        self.patch_size = getattr(self.model.config, "patch_size", 16)  # 16 for ViT-B/16
        self.img_size = 16 * self.patch_size                            # 256 px
        self.project_dir = Path(project_dir)
        self.feat_dir = self.project_dir / "features_mae_b16"
        self.bank_dir = self.project_dir / "featurebanks" / "mae_b16"
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        self.bank_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def extract(self, batch_img_tensor: torch.Tensor) -> torch.Tensor:
        """
        - Uses HF AutoImageProcessor for resizing/normalization to 256x256
        - Disables MAE masking (bool_masked_pos=None) → all 256 patches are visible
        - Returns normalized global features (mean over all patch embeddings)
        """
        # Convert batch of CHW tensors -> list of HWC numpy arrays in [0,1]
        imgs_np = [(img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()) for img in batch_img_tensor]

        # Processor handles resize & normalization (no center crop, fixed 256x256)
        inputs = self.processor(
            images=imgs_np,
            return_tensors="pt",
            do_center_crop=False,
            size={"height": self.img_size, "width": self.img_size},
        ).to(DEVICE)

        # Disable masking: bool_masked_pos=None ensures all patches are processed
        outputs = self.model(pixel_values=inputs["pixel_values"], bool_masked_pos=None)
        hidden = outputs.last_hidden_state  # [B, 1+P, 768] (CLS + patches)

        # Global feature = mean over all patch embeddings (excluding CLS)
        patch_tokens = hidden[:, 1:, :]     # [B, 256, 768]
        feats = patch_tokens.mean(dim=1)    # [B, 768]

        return F.normalize(feats, dim=1)

    def run_all(self, cat: str, data_dir: Path, batch_size=128, overwrite=False):
        out_npz = self.feat_dir / f"{cat}_mae_b16.npz"
        out_csv = self.feat_dir / f"{cat}_meta.csv"
        if not overwrite and out_npz.exists():
            return

        # For MAE we skip torchvision transforms -> processor does resize/normalize
        paths, labels, cats, splits = collect_images(data_dir / cat, cat)
        ds = ImagePathDataset(paths, labels, cats, splits, transform=None)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_keep_meta)
        feats, metas = [], []

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            for imgs, metas_batch in tqdm(dl, desc=f"[MAE ALL] {cat}"):
                imgs = imgs.to(DEVICE)
                f = self.extract(imgs).cpu().numpy()
                feats.append(f)
                metas.extend(metas_batch)

        np.savez_compressed(out_npz, features=np.concatenate(feats, axis=0))
        pd.DataFrame(metas).to_csv(out_csv, index=False)

    def run_bank(self, cat: str, data_dir: Path, batch_size=128, overwrite=False):
        out_npz = self.bank_dir / f"{cat}_bank_mae_b16.npz"
        out_csv = self.bank_dir / f"{cat}_bank_meta.csv"
        if not overwrite and out_npz.exists():
            return

        good_dir = data_dir / cat / "train" / "good"
        paths = sorted(good_dir.rglob("*.png"))
        labels = ["good"] * len(paths)
        cats = [cat] * len(paths)
        splits = ["train"] * len(paths)

        # Again: no torchvision transforms, processor handles everything
        ds = ImagePathDataset(paths, labels, cats, splits, transform=None)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_keep_meta)
        feats, metas = [], []

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            for imgs, metas_batch in tqdm(dl, desc=f"[MAE BANK] {cat}"):
                imgs = imgs.to(DEVICE)
                f = self.extract(imgs).cpu().numpy()
                feats.append(f)
                metas.extend(metas_batch)

        np.savez_compressed(out_npz, features=np.concatenate(feats, axis=0))
        pd.DataFrame(metas).to_csv(out_csv, index=False)


# ========== FACTORY ==========
def get_feature_extractor(name: str, project_dir=Path("/content/drive/MyDrive")):
    if name.lower() == "dino":
        return DinoFeatureExtractor(project_dir=project_dir)
    elif name.lower() == "mae":
        return MAEFeatureExtractor(project_dir=project_dir)
    else:
        raise ValueError(f"Unknown extractor: {name}")



