@echo off
REM Check for Python 3.10 at the specified path
set "PYTHON310_PATH=C:\Users\onuru\AppData\Local\Programs\Python\Python310"

REM Check for Python 3.10 at the specified path using the variable
if exist "%PYTHON310_PATH%\python.exe" (
    echo Python 3.10 found at specified path, using Python 3.10
    set "PYTHON_EXEC=%PYTHON310_PATH%\python.exe"
) else (
    echo Python 3.10 not found at specified path, using default python interpreter
    set "PYTHON_EXEC=python"
)

echo Using Python interpreter: %PYTHON_EXEC%
%PYTHON_EXEC% --version

REM Create a virtual environment in the 'venv' directory
%PYTHON_EXEC% -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install the required packages from requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

echo.
echo Setup complete. The virtual environment is activated.
echo Press any key to exit...
pause >nul

REM Keep the window open
cmd /k