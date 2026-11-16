@echo off
REM OdontoSign confirmatory test with the clinician boxes (about 10 hours)
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\clinician_boxes.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo START %DATE% %TIME% >> %LOG%
echo Running the clinician-box analysis (about 10 hours). Keep the laptop plugged in. Log: %LOG%
python step13_clinician_boxes.py >> %LOG% 2>&1 && (echo END %DATE% %TIME% OK >> %LOG%) || (echo FAILED %DATE% %TIME% >> %LOG%)
echo Finished. See %LOG%
pause
