@echo off
REM Check for venv
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Check for latexmk (basic check)
where latexmk >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: latexmk not found in PATH. PDF generation may fail.
    echo Please install MiKTeX or TeX Live.
    pause
)

echo Starting Resume Fine-Tuner...
streamlit run app.py
pause
