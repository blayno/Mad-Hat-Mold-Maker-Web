@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.11 or 3.12 and enable Add Python to PATH.
  pause
  exit /b 1
)
echo Installing or checking Mad Hat web app dependencies...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency installation failed. Check your Python installation and internet connection.
  pause
  exit /b 1
)
echo.
echo Starting Mad Hat Negative Mold Generator...
start "Mad Hat Browser" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:5000"
py app.py
pause
