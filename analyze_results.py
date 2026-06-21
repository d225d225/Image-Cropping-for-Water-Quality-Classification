"""
analyze_results.py
訓練完成後執行，產出完整結果分析報告。
"""
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from ultralytics import YOLO

# ── 設定 ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/classify/water_quality/weights/best.pt"
DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LABEL_ZH = {"clean": "乾淨", "turbid": "混濁", "dirty": "髒"}
LABEL_EN  = ["clean", "dirty", "turbid"]  # alphabetical

# ── 載入模型 ──────────────────────────────────────────────────────────────────
def load_model():
    model = YOLO(MODEL_PATH)
    print(f"✅ 模型載入：{MODEL_PATH}")
    return model

# ── 在 test 集上推論 ──────────────────────────────────────────────────────────
def run_inference(model):
    y_true, y_pred, y_conf, paths = [], [], [], []
    labels = sorted([d.name for d in (DATASET_DIR / "test").iterdir() if d.is_dir()])

    for label in labels:
        imgs = list((DATASET_DIR / "test" / label).glob("*"))
        for img_path in imgs:
            r    = model(str(img_path))[0]
            pred = r.names[r.probs.top1]
            conf = r.probs.top1conf.item()
            y_true.append(label)
            y_pred.append(pred)
            y_conf.append(conf)
            paths.append(img_path)

    return labels, y_true, y_pred, y_conf, paths

# ── 混淆矩陣 ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, y_true, y_pred):
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{l}\n({LABEL_ZH[l]})" for l in labels], fontsize=11)
    ax.set_yticklabels([f"{l}\n({LABEL_ZH[l]})" for l in labels], fontsize=11)
    ax.set_xlabel("預測類別", fontsize=12)
    ax.set_ylabel("真實類別", fontsize=12)
    ax.set_title("混淆矩陣（Confusion Matrix）", fontsize=14)

    for i in range(n):
        for j in range(n):
            color = "white" if cm[i][j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    fontsize=14, color=color, fontweight="bold")

    plt.tight_layout()
    path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ 混淆矩陣已儲存：{path}")
    return cm, labels

# ── Precision / Recall / F1 ───────────────────────────────────────────────────
def calc_metrics(cm, labels):
    metrics = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    accuracy = sum(cm[i][i] for i in range(len(labels))) / cm.sum()
    return metrics, accuracy

# ── 誤判案例 ──────────────────────────────────────────────────────────────────
def plot_wrong_cases(y_true, y_pred, y_conf, paths, max_cases=9):
    wrong = [(t, p, c, path)
             for t, p, c, path in zip(y_true, y_pred, y_conf, paths)
             if t != p]

    if not wrong:
        print("🎉 測試集全部預測正確，沒有誤判案例！")
        return

    sample = random.sample(wrong, min(max_cases, len(wrong)))
    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    from PIL import Image
    for ax, (true, pred, conf, path) in zip(axes, sample):
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "無法讀取", ha="center", va="center")
        ax.set_title(
            f"真實：{LABEL_ZH[true]}\n"
            f"預測：{LABEL_ZH[pred]}（{conf:.1%}）",
            fontsize=10, color="red"
        )
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.suptitle("模型誤判案例", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / "wrong_cases.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ 誤判案例已儲存：{path}（共 {len(wrong)} 個誤判，顯示 {len(sample)} 個）")

# ── 儲存文字報告 ──────────────────────────────────────────────────────────────
def save_report(metrics, accuracy, y_true, y_pred, labels):
    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))

    lines = [
        "# 訓練結果分析報告",
        "",
        "## 整體準確率",
        f"- **Accuracy：{accuracy:.1%}**（{correct}/{total} 張預測正確）",
        "",
        "## 各類別指標",
        "",
        "| 類別 | 中文 | Precision | Recall | F1-Score |",
        "|------|------|-----------|--------|----------|",
    ]
    for label in labels:
        m = metrics[label]
        lines.append(
            f"| {label} | {LABEL_ZH[label]} "
            f"| {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} |"
        )

    wrong = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    from collections import Counter
    wrong_pairs = Counter(wrong)
    lines += [
        "",
        "## 混淆矩陣解讀",
        "",
        "模型最常見的誤判：",
    ]
    for (t, p), count in wrong_pairs.most_common(3):
        lines.append(f"- 把「{LABEL_ZH[t]}」誤判成「{LABEL_ZH[p]}」：{count} 次")

    lines += [
        "",
        "## 給高中生看的結果摘要",
        "",
        f"這個 AI 模型辨識瓦磘溝水質的準確率是 **{accuracy:.1%}**。",
        "",
        "簡單來說：",
        f"- 每 100 張照片，模型大約能正確判斷 {accuracy*100:.0f} 張",
        "- 最容易搞混的是「混濁」和「髒」的水——因為兩者顏色接近，",
        "  連人眼有時候也很難區分",
        "- 「乾淨」的水通常比較好辨識，因為顏色明顯比較清澈",
        "",
        "## 可以怎麼改善？",
        "",
        "1. **增加照片數量**：每類至少 100 張，尤其是最少的類別",
        "2. **統一拍攝條件**：相同時間、光線、角度，減少環境干擾",
        "3. **嘗試更大的模型**：`yolov8m-cls` 或 `yolov8l-cls` 準確率更高",
        "4. **資料增強**：旋轉、翻轉、調整亮度，讓模型學到更多變化",
    ]

    report_path = RESULTS_DIR / "analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 文字報告已儲存：{report_path}")

    # 也印出來
    print("\n" + "="*50)
    print(f"整體準確率：{accuracy:.1%}")
    print(f"{'類別':10s} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 44)
    for label in labels:
        m = metrics[label]
        print(f"{LABEL_ZH[label]:10s} {m['precision']:>10.1%} "
              f"{m['recall']:>10.1%} {m['f1']:>10.1%}")

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model()
    labels, y_true, y_pred, y_conf, paths = run_inference(model)
    cm, labels = plot_confusion_matrix(labels, y_true, y_pred)
    metrics, accuracy = calc_metrics(cm, labels)
    plot_wrong_cases(y_true, y_pred, y_conf, paths)
    save_report(metrics, accuracy, y_true, y_pred, labels)
    print(f"\n📁 所有結果已儲存至：{RESULTS_DIR.resolve()}")
