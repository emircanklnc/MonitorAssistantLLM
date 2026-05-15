@echo off
title Monitor Asistani
echo ========================================
echo    Monitor Asistani Baslatiliyor...
echo ========================================
echo.
cd /d "%~dp0"
streamlit run app.py
pause
