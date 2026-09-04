@echo off
setlocal
cd /d "%~dp0"

rem A .venv copied from another machine still has python.exe but points at a
rem Python that is not here, so test that it actually runs before trusting it.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Existing .venv is broken ^(built for another machine^); recreating...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 --version >nul 2>&1
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 goto :error
)

echo Installing required packages...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

echo Setup completed.
exit /b 0

:error
echo Setup failed.
exit /b 1
