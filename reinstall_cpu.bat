@echo off
cd /d "%~dp0"
title EmbeddingGemma 2 - CPU mode
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launcher.ps1" -Reinstall -ForceCPU
pause
