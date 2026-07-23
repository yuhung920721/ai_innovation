# AI輔助居家復健訓練系統

以 MediaPipe Pose 骨架偵測技術，即時判斷上肢復健動作的完成度與標準度，取代原本以 Teachable Machine 訓練的影像分類模型，強化「感知－決策－行動」的 Physical AI 系統架構。

---

## 專案背景

本專案原為大學實務專題《即時影像技術輔助上肢復健》，使用 Teachable Machine 訓練動作分類模型，搭配 OpenCV 讀取攝影機影像進行復健動作辨識與計數。此次技術升級將辨識核心從「影像分類」改為「骨架關節角度計算」：

- **感知**：MediaPipe Pose 擷取全身關節座標
- **決策**：計算關節角度、判斷動作正確性與完成度
- **行動**：即時視覺回饋（燈號、進度條、分數）

---

## 支援動作

| 動作 | 程式檔案 | 復健部位 |
|---|---|---|
| 手臂側舉（肩關節外展） | `shoulder_abduction.py` | 肩關節活動度 |
| 手臂前舉（肩關節前屈） | `shoulder_flexion.py` | 肩關節活動度 |
| 手臂平舉（肩關節水平內收） | `shoulder_horizontal_adduction.py` | 肩後側關節囊活動度 |
| 手臂後舉（肩後伸暨觸頸伸展） | `shoulder_extension_stretch.py` | 肩關節外旋活動度 |

每個動作皆遵循相同的三階段流程：**準備動作(S0) → 動作一(S1) → 動作二(S2) → 回到準備動作 → 次數+1**，並提供即時燈號回饋、完成度百分比、動作標準度分數。

---

## 專案結構

```
ai_innovation/
├── pose_utils.py                        # 共用模組：MediaPipe初始化、角度計算、計數器狀態機
├── shoulder_abduction.py                # 手臂側舉
├── shoulder_flexion.py                  # 手臂前舉
├── shoulder_horizontal_adduction.py     # 手臂平舉
├── shoulder_extension_stretch.py        # 手臂後舉
├── calibration_tool.py                  # 角度校正工具（測試/記錄實際角度數據用）
├── requirements.txt                     # Python套件需求
├── NotoSansTC-Regular.ttf               # 中文字型檔（畫面文字顯示用）
└── docs/
    ├── shoulder_abduction_角度測量與評分說明.md
    ├── shoulder_flexion_角度測量說明.md
    ├── shoulder_horizontal_adduction_角度測量說明.md
    ├── shoulder_extension_stretch_角度測量說明.md
    └── 復健姿勢評分說明.md              # 四動作共用評分機制說明
```

---

## 環境需求

- Python 3.9 ~ 3.12（**不支援 3.13**，MediaPipe 尚無穩定支援）
- Webcam

---

## 安裝步驟

```bash
# 1. 建立虛擬環境
python -m venv venv

# 2. 啟用虛擬環境
# Windows PowerShell
venv\Scripts\Activate.ps1
# Mac / Linux
source venv/bin/activate

# 3. 安裝套件
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 使用方式

### 執行單一動作偵測

```bash
python shoulder_abduction.py              # 預設偵測右手
python shoulder_abduction.py --side left  # 偵測左手
```

四支動作程式皆支援 `--side left` / `--side right` 參數。

操作按鍵：
- `q` 或 `ESC`：結束程式並顯示總得分

### 角度校正工具

```bash
python calibration_tool.py
```

即時顯示關節角度數值，並採用「回到準備動作才開始記錄下一次」的重置機制，方便錄影測試、記錄真實角度數據以校正各動作的判斷門檻。

- 按 `r`：清空目前記錄
- 按 `q` / `ESC`：結束並輸出統計摘要

---

## 技術文件

各動作詳細的角度計算方式與判斷邏輯，請參考 `docs/` 資料夾中對應的說明文件；四個動作共用的評分機制，請參考《復健姿勢評分說明》。

---

## 待辦與已知限制

- 手臂後舉動作因手部末端動作方向為前後方向（矢狀面），正面拍攝時手部在動作末端容易被頭部遮擋，**建議側身面對鏡頭**進行此動作。
- 目前角度門檻依團隊成員實測數據校正，若使用者體型/柔軟度差異較大，建議搭配 `calibration_tool.py` 個別微調。
- 網頁前端與 Flask 路由整合由另一位組員負責，本文件僅涵蓋動作偵測核心邏輯。