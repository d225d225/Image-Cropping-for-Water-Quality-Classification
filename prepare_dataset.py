"""
prepare_dataset.py
從 Google Drive 下載裁切後的水質照片，整理成 YOLOv8-cls 訓練格式。
資料切分比例：train 70% / val 20% / test 10%
"""
import os
import shutil
import random
from pathlib import Path
import gdown

# ── 設定 ─────────────────────────────────────────────────────────────────────
FOLDER_ID   = "12OOjkS7GilRcvaVafh7NJKWk70mA1pDD"
RAW_DIR     = Path("raw_data")
DATASET_DIR = Path("dataset")
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
# TEST_RATIO  = 0.1（剩餘）
SEED        = 42

LABEL_MAP = {
    "1": "dirty",    # 髒
    "3": "turbid",   # 混濁
    "5": "clean",    # 乾淨
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

# ── 下載 ──────────────────────────────────────────────────────────────────────
def download():
    print("📥 從 Google Drive 下載資料集...")
    gdown.download_folder(
        f"https://drive.google.com/drive/folders/{FOLDER_ID}",
        output=str(RAW_DIR),
        quiet=False,
        use_cookies=False,
    )
    print("✅ 下載完成\n")

# ── 檢查異常 ──────────────────────────────────────────────────────────────────
def check_anomalies(images: list) -> list:
    warnings = []
    seen = {}
    for img in images:
        size = img.stat().st_size
        if size < 5000:
            warnings.append(f"⚠️  疑似損毀（{size} bytes）：{img.name}")
        key = size
        if key in seen:
            warnings.append(f"⚠️  可能重複：{img.name} 與 {seen[key].name}")
        else:
            seen[key] = img
    return warnings

# ── 整理資料集 ────────────────────────────────────────────────────────────────
def build_dataset():
    random.seed(SEED)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val", "test"]:
        for label in LABEL_MAP.values():
            (DATASET_DIR / split / label).mkdir(parents=True, exist_ok=True)

    print("📊 資料集切分（70% train / 20% val / 10% test）：\n")

    for folder_name, label in LABEL_MAP.items():
        src_dir = RAW_DIR / folder_name
        if not src_dir.exists():
            matches = [d for d in RAW_DIR.iterdir()
                       if d.is_dir() and folder_name in d.name]
            src_dir = matches[0] if matches else None

        if not src_dir:
            print(f"✗ 找不到資料夾 {folder_name}，跳過")
            continue

        images = sorted([f for f in src_dir.rglob("*")
                         if f.is_file() and f.suffix.lower() in IMG_EXTS])

        # 異常檢查
        warnings = check_anomalies(images)
        for w in warnings:
            print(w)

        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():
            for i, img in enumerate(split_imgs):
                ext = img.suffix.lower() if img.suffix else ".jpg"
                shutil.copy(img, DATASET_DIR / split_name / label / f"{label}_{i:04d}{ext}")

        print(f"  {label:8s}（{folder_name}）："
              f"train {len(splits['train']):3d} / "
              f"val {len(splits['val']):3d} / "
              f"test {len(splits['test']):3d}  "
              f"（共 {n} 張）")

    total = sum(1 for _ in DATASET_DIR.rglob("*") if _.is_file())
    print(f"\n✅ 完成！資料集共 {total} 張，路徑：{DATASET_DIR.resolve()}")

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not RAW_DIR.exists():
        download()
    else:
        print(f"ℹ️  已有 {RAW_DIR}，跳過下載（刪除該資料夾可重新下載）")
    build_dataset()
