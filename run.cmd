@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import cv2, numpy, PIL, pymupdf" >nul 2>&1
    if not errorlevel 1 goto :ready
)
call setup.cmd
if errorlevel 1 goto :error

:ready
echo Extracting lecture slides...
echo.
".venv\Scripts\python.exe" lecture_video_to_pdf.py ".." --output "..\pdf_output" %*
if errorlevel 1 goto :error

echo.
echo Finished.  PDFs are in the pdf_output folder next to your videos.
echo To keep the handwriting too, run:  run.cmd --ink-mode epochs
pause
exit /b 0

:error
echo.
echo Extraction failed.  Review the message above.
pause
exit /b 1
