# 專案提示詞紀錄

> 這份文件記錄本專案的完整提示詞，供教學參考使用。

---

## 使用者提示詞（繁體中文完整版）

```
我要做一個瓦磘溝水質影像辨識專案，請你協助我完成「資料整理 → 建立 GitHub repo
→ 在 Google Colab 用 YOLOv8 訓練分類模型 → 分析結果 → 撰寫 README」全部流程。
請全程用繁體中文跟我說明你在做什麼，因為我會把過程也用在教學上。

【背景】
我帶領的「瓦磘溝行動研究社」學生實際去瓦磘溝拍攝照片，並依水質狀況分類成三類。
照片已經裁切完成，只保留水的部分，不需要再做裁切處理。
照片存放在 Google 雲端硬碟：
https://drive.google.com/drive/folders/12OOjkS7GilRcvaVafh7NJKWk70mA1pDD

資料夾結構：
- 5（乾淨）→ class: clean
- 3（混濁）→ class: turbid
- 1（髒）  → class: dirty

【任務一：資料下載與整理】
1. 協助下載三個子資料夾的照片，列出每個資料夾的照片數量、檔案格式、解析度範圍，
   並抽幾張縮圖讓我確認下載的內容正確。
2. 依照 clean / turbid / dirty 三類整理成標準資料夾結構，
   並切分 train / val / test（建議比例 7:2:1）。
3. 如果發現照片有明顯異常（損毀、解析度極端不一致、重複），列出來提醒，不要自己刪除。

【任務二：建立 GitHub Repo】
1. 在 d225d225 帳號建立 public repo：Image-Cropping-for-Water-Quality-Classification
2. 放上：資料整理程式碼、Colab notebook、模型權重（best.pt）、訓練結果分析圖
3. 完成完整 README.md（含專案背景、資料說明、YOLOv8 選用原因、Colab 一鍵開啟徽章、
   結果分析、改善方向）

【任務三：用 YOLOv8 在 Google Colab 訓練】
- 使用 yolov8s-cls（影像分類，非物件偵測）
- Colab notebook 分段清楚，不需要寫程式、照順序執行就能跑完
- 適合沒有程式背景的高中生操作

【任務四：訓練結果分析】
1. 整體 accuracy、各類別 precision / recall / F1
2. 混淆矩陣 + 白話文解釋
3. 誤判案例照片
4. loss / accuracy 曲線圖
5. 給高中生看的結果摘要
```

---

## AI 回應策略說明

本專案使用 **Claude Code（Sonnet 4.6）** 完成，採用以下策略：

### 為什麼選 YOLOv8 分類模型而非偵測模型？

| 比較項目 | 物件偵測（YOLOv8-det） | 影像分類（YOLOv8-cls） |
|---------|----------------------|----------------------|
| 標註方式 | 需要畫 bounding box | 只需資料夾分類 ✅ |
| 照片前處理 | 需找出水面位置 | 照片已裁切好 ✅ |
| 結果呈現 | 框框 + 類別 | 類別 + 機率 ✅ 更直觀 |
| 適合教學 | 需解釋 IoU 等概念 | 準確率、混淆矩陣 ✅ |

### 訓練參數選擇邏輯

- `yolov8s-cls`：small 版本，比 nano 準但比 medium 快，適合資料量 < 500 張的場景
- `imgsz=224`：YOLOv8-cls 預設輸入尺寸，與 ImageNet 預訓練一致
- `epochs=100`：搭配 `patience=20` 早停，避免過擬合
- `batch=16`：Colab T4 GPU 記憶體足夠，訓練速度快
