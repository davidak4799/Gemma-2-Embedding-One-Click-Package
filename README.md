# Google EmbeddingGemma 2 本地多模態搜索 · Windows 11 一鍵啟動包

用 [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) 喺本機做 **文字 / 圖片 / 影片 / 音頻** 跨模態語義搜索。全部本地運行，檔案唔會上傳。

## 快速開始
1. 下載本 repo（綠色 Code 掣 → Download ZIP），解壓到純英文路徑，例如 `D:\EmbeddingGemma2`
2. 雙擊 `start.bat`
3. 第一次會自動安裝（約 10–30 分鐘，需約 6GB 空間），之後瀏覽器自動打開 `http://localhost:8501`

## 自動處理
- 免安裝版 [uv](https://github.com/astral-sh/uv) 安裝 Python 3.12，全部放喺資料夾內，唔影響系統
- 按 NVIDIA 驅動自動揀 PyTorch：≥580 → cu130，≥572 → cu128，≥561 → cu126，其他 → CPU
- GPU 用 bfloat16，CPU 用 float32（模型唔支援 float16）
- 下載失敗自動轉清華 PyPI 鏡像 / hf-mirror.com

## 檔案
| 檔案 | 用途 |
|---|---|
| `start.bat` | 一鍵安裝 + 啟動 |
| `reinstall.bat` | 重新安裝環境（保留模型同索引） |
| `reinstall_cpu.bat` | 強制 CPU 版重裝 |
| `scripts/launcher.ps1` | 安裝 / 啟動邏輯 |
| `app.py` | Streamlit 搜索介面 |
| `requirements.txt` | Python 依賴 |

已有模型檔案可直接放入 `models/embeddinggemma-2/`，免再下載。

## 注意
- 唔好索引成個 C 盤
- CPU 建立圖片 / 影片索引會好慢，建議用 NVIDIA 顯卡
- 音頻 / 影片輸入格式已寫後備方案，但未經 Windows 實測，歡迎開 issue 回報
