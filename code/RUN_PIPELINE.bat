@echo off
REM Double-click to run the whole pipeline. Output is written to logs\run.log
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\run.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

echo ================================================================ >> %LOG%
echo START %DATE% %TIME% >> %LOG%

REM ---- find a Python that has torch -----------------------------------------
set PY=
for %%P in ("python" "py -3.12" "py" "C:\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe") do (
    if not defined PY (
        %%~P -c "import torch" >nul 2>&1 && set PY=%%~P
    )
)
if not defined PY (
    echo ERROR: no Python with torch found on PATH >> %LOG%
    echo ERROR: no Python with torch found on PATH
    pause
    exit /b 1
)
echo Using: %PY% >> %LOG%
%PY% -c "import sys, torch; print(sys.executable, torch.__version__, 'xpu', hasattr(torch,'xpu') and torch.xpu.is_available())" >> %LOG% 2>&1

echo Installing requirements... (see %LOG%)
%PY% -m pip install -q -r requirements.txt >> %LOG% 2>&1

REM ---- pipeline --------------------------------------------------------------
echo Running pipeline... this takes hours. Progress is in %LOG%
%PY% run_all.py --skip-perm %* >> %LOG% 2>&1
echo END %DATE% %TIME% exit=%ERRORLEVEL% >> %LOG%
echo Finished. See %LOG%
pause
