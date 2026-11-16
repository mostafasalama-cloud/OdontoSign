@echo off
REM CBCT-centred study: features (step 6), then per task 3D CNN (8), models (7), stats/report (9)
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\cbct_study.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PY=python
echo START %DATE% %TIME% >> %LOG%
echo [1/3] Lesion features (about 1-2 h)... see %LOG%
%PY% step6_cbct_features.py >> %LOG% 2>&1 || goto :fail
for %%T in (ALL4 AME_vs_DC) do (
    echo [2/3] Task %%T: 3D CNN, models, statistics... see %LOG%
    set DOLCHID_TASK=%%T
    %PY% step8_cbct_3dcnn.py >> %LOG% 2>&1 || goto :fail
    %PY% step7_cbct_evaluate.py >> %LOG% 2>&1 || goto :fail
    %PY% step9_cbct_report.py >> %LOG% 2>&1 || goto :fail
)
echo END %DATE% %TIME% OK >> %LOG%
echo Finished. See %LOG%
pause
exit /b 0
:fail
echo FAILED %DATE% %TIME% >> %LOG%
echo FAILED - see %LOG%
pause
exit /b 1
