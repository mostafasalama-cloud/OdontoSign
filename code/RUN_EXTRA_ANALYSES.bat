@echo off
REM Additional analyses: mask robustness, interpretability, calibration, decision curves
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\extra_analyses.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo START %DATE% %TIME% >> %LOG%
echo Running additional analyses (about 1-2 h)... see %LOG%
python step10_cbct_extra.py >> %LOG% 2>&1 && (echo END %DATE% %TIME% OK >> %LOG%) || (echo FAILED %DATE% %TIME% >> %LOG%)
echo Finished. See %LOG%
pause
