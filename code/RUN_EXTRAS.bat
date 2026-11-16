@echo off
REM Permutation tests (primary task) + 4-class and original-100 sensitivity analyses
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\extras.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PY=python
echo START %DATE% %TIME% >> %LOG%
echo [1/3] Permutation tests... see %LOG%
%PY% step4_statistics.py >> %LOG% 2>&1
%PY% step5_report.py >> %LOG% 2>&1
echo [2/3] Four-class task... see %LOG%
%PY% run_all.py --task ALL4 --skip-perm >> %LOG% 2>&1
echo [3/3] Original 100-patient subset... see %LOG%
%PY% run_all.py --original100 --skip-perm >> %LOG% 2>&1
echo END %DATE% %TIME% exit=%ERRORLEVEL% >> %LOG%
echo Finished. See %LOG%
pause
