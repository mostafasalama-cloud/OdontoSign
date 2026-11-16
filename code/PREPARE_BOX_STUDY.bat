@echo off
REM Prepares the 60 anonymised, downsampled CBCT volumes for the clinician box-drawing study
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\box_study.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo START %DATE% %TIME% >> %LOG%
echo Preparing 60 cases for the clinician (about 10-20 minutes)... see %LOG%
python prepare_box_study.py >> %LOG% 2>&1 && (echo END %DATE% %TIME% OK >> %LOG%) || (echo FAILED %DATE% %TIME% >> %LOG%)
echo Finished. See %LOG%
pause
