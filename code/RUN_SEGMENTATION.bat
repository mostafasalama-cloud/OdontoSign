@echo off
REM Semi-automatic pipeline: box prompt -> 3D U-Net (5-fold, on GPU) -> descriptors -> classification
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\segmentation.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo START %DATE% %TIME% >> %LOG%
echo Running semi-automatic segmentation (several hours, resumable)... see %LOG%
python step12_semiauto_seg.py >> %LOG% 2>&1 && (echo END %DATE% %TIME% OK >> %LOG%) || (echo FAILED %DATE% %TIME% >> %LOG%)
echo Finished. See %LOG%
pause
