Google EmbeddingGemma 2 本地多模態搜索 一鍵包（Windows 11）
==========================================================

【用法】
1. 將成個資料夾解壓到純英文路徑，例如 D:\EmbeddingGemma2
2. 雙擊 start.bat
   - 第一次會自動：下載 uv → 安裝 Python 3.12 → 偵測顯卡 → 安裝對應 PyTorch → 安裝依賴 → 下載模型（約 1.5GB）
   - 全程約 10-30 分鐘（睇網速），需要約 6GB 硬碟空間
   - 之後再雙擊就會直接啟動，瀏覽器自動打開 http://localhost:8501
3. 喺左邊輸入資料夾路徑 → 「開始建立 / 更新索引」→ 喺右邊用文字 / 圖片 / 音頻 / 影片搜索

【顯卡自動匹配】
- NVIDIA 驅動 ≥ 580 → CUDA 13.0（RTX 50 系列建議用呢個）
- NVIDIA 驅動 ≥ 572 → CUDA 12.8
- NVIDIA 驅動 ≥ 561 → CUDA 12.6
- 驅動太舊 / AMD / Intel 顯示卡 / 冇獨立顯卡 → CPU 模式
- GPU 自動用 bfloat16，CPU 用 float32（模型唔支援 float16）

【其他檔案】
- reinstall.bat      ：環境壞咗 / 更新咗顯卡驅動後，重新安裝（保留模型同索引）
- reinstall_cpu.bat  ：強制用 CPU 版重新安裝（GPU 版出問題時用）
- install_log.txt    ：安裝記錄，出錯時睇呢個
- models\            ：模型檔案。如果你已經用網盤下載咗模型，可以將 config.json、model.safetensors 等全部檔案放入 models\embeddinggemma-2\ ，就唔使再下載
- index\             ：索引資料，刪除即清空所有索引

【注意】
- 唔好揀成個 C 盤做索引，系統檔太多會好慢
- 圖片 / 影片 / 音頻用 CPU 建索引會非常慢，有 NVIDIA 顯卡會快好多；建議先揀細資料夾試
- 再按一次「建立 / 更新索引」只會處理新增或改動過嘅檔案
- 支援文字類：txt md csv json pdf docx pptx xlsx 及常見程式碼
  圖片：jpg png webp bmp gif tif；音頻：mp3 wav flac m4a aac ogg wma opus
  影片：mp4 mkv mov avi webm wmv flv m4v 3gp（每條均勻抽 16 幀；音頻取頭 5 分鐘）
- 所有運算都喺本機完成，檔案唔會上傳
