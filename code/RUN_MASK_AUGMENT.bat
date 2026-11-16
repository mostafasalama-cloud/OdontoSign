@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\mask_augment.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo START %DATE% %TIME% >> %LOG%
echo Running mask-augmentation analysis (about 10-20 min)... see %LOG%
python step11_mask_augment.py >> %LOG% 2>&1 && (echo END %DATE% %TIME% OK >> %LOG%) || (echo FAILED %DATE% %TIME% >> %LOG%)
echo Finished. See %LOG%
pause
