@echo off
cd /d "%~dp0"
title EmbeddingGemma 2 Local Search
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launcher.ps1"
if errorlevel 1 pause
