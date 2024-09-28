@echo off
REM Activate the virtual environment
call venv\Scripts\activate

REM Run the Python script
REM 

echo.
echo Script execution complete. The virtual environment is still activated.
echo Press any key to exit...
pause >nul

REM Keep the window open
cmd /k