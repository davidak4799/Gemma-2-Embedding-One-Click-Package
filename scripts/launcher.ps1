# EmbeddingGemma 2 一鍵啟動包 - Windows 11
# 第一次運行：自動安裝 uv + Python 3.12 + PyTorch(自動匹配顯卡) + 依賴 + 模型
# 之後運行：直接啟動 http://localhost:8501
param(
    [switch]$Reinstall,
    [switch]$ForceCPU
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$Root      = Split-Path -Parent $PSScriptRoot
$Runtime   = Join-Path $Root "runtime"
$UvDir     = Join-Path $Runtime "uv"
$UvExe     = Join-Path $UvDir "uv.exe"
$Venv      = Join-Path $Root "venv"
$Py        = Join-Path $Venv "Scripts\python.exe"
$ModelDir  = Join-Path $Root "models\embeddinggemma-2"
$Marker    = Join-Path $Runtime "installed.txt"
$LogFile   = Join-Path $Root "install_log.txt"
$ModelRepo = "google/embeddinggemma-2"

# 全部放喺本資料夾內，唔污染系統
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime "python"
$env:UV_CACHE_DIR          = Join-Path $Runtime "uv-cache"
$env:UV_LINK_MODE          = "copy"
$env:HF_HOME               = Join-Path $Runtime "hf-cache"
$env:PYTHONUTF8            = "1"
$env:PYTHONIOENCODING      = "utf-8"

function Say($msg, $color = "Cyan") { Write-Host $msg -ForegroundColor $color }
function Fail($msg) {
    Write-Host ""
    Write-Host "[錯誤] $msg" -ForegroundColor Red
    Write-Host "詳細記錄請睇：$LogFile" -ForegroundColor Yellow
    exit 1
}
function Run($exe, [string[]]$argList) {
    $ErrorActionPreference = "Continue"   # 避免 PowerShell 5.1 將 stderr 當錯誤中斷
    Add-Content -Path $LogFile -Value ("`n>>> " + $exe + " " + ($argList -join " ")) -Encoding UTF8
    & $exe @argList 2>&1 | ForEach-Object { $line = "$_"; Add-Content -Path $LogFile -Value $line -Encoding UTF8; Write-Host $line }
    return $LASTEXITCODE
}

Write-Host ""
Say "=================================================" "Green"
Say "   Google EmbeddingGemma 2 本地多模態搜索 一鍵包" "Green"
Say "   文字 / 圖片 / 影片 / 音頻 互相搜索" "Green"
Say "=================================================" "Green"
Write-Host ""

if ($Root -match '[^\x00-\x7F]') {
    Say "[提示] 資料夾路徑含中文或特殊字元：$Root" "Yellow"
    Say "       建議放喺純英文路徑，例如 D:\EmbeddingGemma2，可避免部分套件出錯。" "Yellow"
    Write-Host ""
}

if (-not [Environment]::Is64BitOperatingSystem) { Fail "需要 64 位元 Windows。" }

if ($Reinstall) {
    Say "正在清除舊環境（模型同索引會保留）..." "Yellow"
    if (Test-Path $Venv)   { Remove-Item $Venv -Recurse -Force }
    if (Test-Path $Marker) { Remove-Item $Marker -Force }
}

New-Item -ItemType Directory -Force -Path $Runtime, $UvDir | Out-Null

# ---------------- 安裝 ----------------
if (-not (Test-Path $Marker) -or -not (Test-Path $Py)) {
    Set-Content -Path $LogFile -Value ("Install log " + (Get-Date)) -Encoding UTF8

    # 1. uv（Python 安裝/套件管理工具，比 pip 快好多）
    if (-not (Test-Path $UvExe)) {
        Say "[1/5] 下載 uv 工具..."
        $zip = Join-Path $Runtime "uv.zip"
        $urls = @(
            "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip",
            "https://ghfast.top/https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        )
        $ok = $false
        foreach ($u in $urls) {
            try { Invoke-WebRequest -Uri $u -OutFile $zip -UseBasicParsing -TimeoutSec 300; $ok = $true; break }
            catch { Say "   下載失敗，換線路重試..." "Yellow" }
        }
        if (-not $ok) { Fail "下載 uv 失敗，請檢查網絡。" }
        Expand-Archive -Path $zip -DestinationPath $UvDir -Force
        Remove-Item $zip -Force
        $found = Get-ChildItem -Path $UvDir -Recurse -Filter "uv.exe" | Select-Object -First 1
        if ($found -and $found.FullName -ne $UvExe) { Copy-Item $found.FullName $UvExe -Force }
        if (-not (Test-Path $UvExe)) { Fail "搵唔到 uv.exe" }
    } else { Say "[1/5] uv 已存在，略過" "DarkGray" }

    # 2. Python 3.12 + 虛擬環境
    Say "[2/5] 安裝 Python 3.12 及建立虛擬環境 venv..."
    if ((Run $UvExe @("python", "install", "3.12")) -ne 0) { Fail "安裝 Python 3.12 失敗" }
    if (-not (Test-Path $Py)) {
        if ((Run $UvExe @("venv", $Venv, "--python", "3.12", "--seed")) -ne 0) { Fail "建立 venv 失敗" }
    }

    # 3. 偵測顯卡，自動揀 PyTorch 版本
    Say "[3/5] 偵測顯卡..."
    $torchIndex = "https://download.pytorch.org/whl/cpu"
    $mode = "CPU"
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
    if ($ForceCPU) {
        Say "   已指定 CPU 模式" "Yellow"
    } elseif ($smi) {
        try {
            $info = (& $smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
            $parts = $info.Split(",") | ForEach-Object { $_.Trim() }
            $gpuName = $parts[0]; $drv = [double]($parts[1].Split(".")[0] + "." + $parts[1].Split(".")[1]); $vram = $parts[2]
            Say "   顯卡：$gpuName   驅動：$($parts[1])   顯存：$vram" "Green"
            if     ($drv -ge 580) { $torchIndex = "https://download.pytorch.org/whl/cu130"; $mode = "CUDA 13.0" }
            elseif ($drv -ge 572) { $torchIndex = "https://download.pytorch.org/whl/cu128"; $mode = "CUDA 12.8" }
            elseif ($drv -ge 561) { $torchIndex = "https://download.pytorch.org/whl/cu126"; $mode = "CUDA 12.6" }
            else {
                Say "   顯卡驅動太舊（$($parts[1])），請到 NVIDIA 官網更新驅動後執行 reinstall.bat。暫時用 CPU 模式。" "Yellow"
            }
        } catch { Say "   讀取 NVIDIA 資訊失敗，用 CPU 模式" "Yellow" }
    } else {
        $vc = (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name) -join " / "
        Say "   未偵測到 NVIDIA 顯卡（$vc），用 CPU 模式（一樣用得，建索引會慢啲）" "Yellow"
    }
    Say "   PyTorch 模式：$mode" "Green"

    # 4. 安裝 PyTorch + 依賴
    Say "[4/5] 安裝 PyTorch 同依賴套件（第一次約 2-3GB，請耐心等）..."
    $code = Run $UvExe @("pip", "install", "--python", $Py, "torch", "torchvision", "--index-url", $torchIndex)
    if ($code -ne 0 -and $mode -ne "CPU") {
        Say "   GPU 版安裝失敗，改裝 CPU 版..." "Yellow"
        $mode = "CPU"
        $code = Run $UvExe @("pip", "install", "--python", $Py, "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")
    }
    if ($code -ne 0) { Fail "安裝 PyTorch 失敗" }

    $req = Join-Path $Root "requirements.txt"
    $code = Run $UvExe @("pip", "install", "--python", $Py, "-r", $req)
    if ($code -ne 0) {
        Say "   官方源失敗，改用清華鏡像重試..." "Yellow"
        $code = Run $UvExe @("pip", "install", "--python", $Py, "-r", $req, "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple")
    }
    if ($code -ne 0) { Fail "安裝依賴失敗" }

    Say "   驗證 PyTorch..."
    & $Py -c "import torch; print('   torch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

    # 5. 模型
    Say "[5/5] 準備模型 $ModelRepo（約 1.5GB）..."
    if (Test-Path (Join-Path $ModelDir "model.safetensors")) {
        Say "   模型已存在，略過下載" "DarkGray"
    } else {
        $hf = Join-Path $Venv "Scripts\hf.exe"
        $code = Run $hf @("download", $ModelRepo, "--local-dir", $ModelDir)
        if ($code -ne 0) {
            Say "   Hugging Face 官方下載失敗，改用 hf-mirror.com 鏡像..." "Yellow"
            $env:HF_ENDPOINT = "https://hf-mirror.com"
            $code = Run $hf @("download", $ModelRepo, "--local-dir", $ModelDir)
            Remove-Item Env:\HF_ENDPOINT -ErrorAction SilentlyContinue
        }
        if ($code -ne 0) {
            Fail "模型下載失敗。你亦可以手動將模型檔案（config.json、model.safetensors 等）放入：$ModelDir 再雙擊 start.bat"
        }
    }

    Set-Content -Path $Marker -Value ("installed " + (Get-Date) + " mode=" + $mode) -Encoding UTF8
    Say "安裝完成！" "Green"
    Write-Host ""
}

if (-not (Test-Path (Join-Path $ModelDir "model.safetensors"))) {
    Fail "搵唔到模型：$ModelDir 。請執行 reinstall.bat 或手動放入模型檔案。"
}

# ---------------- 啟動 ----------------
$env:EG2_MODEL_DIR       = $ModelDir
$env:EG2_INDEX_DIR       = Join-Path $Root "index"
$env:HF_HUB_OFFLINE      = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$port = 8501
while ((Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { $port++ }

Say "正在啟動本地 AI 搜索 ... 瀏覽器會自動打開 http://localhost:$port" "Green"
Say "（用完直接關閉呢個黑色視窗即可）" "DarkGray"
Write-Host ""
& $Py -m streamlit run (Join-Path $Root "app.py") --server.port $port --server.address 127.0.0.1 --browser.gatherUsageStats false --server.maxUploadSize 2048 --theme.base light
if ($LASTEXITCODE -ne 0) { Fail "程式異常退出" }
