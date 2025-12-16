@echo off
REM Quick start script for Windows

echo ==========================================
echo Linguistically-Aware EWC Experiment
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM Run experiment
echo.
echo Starting experiment...
echo This will train 4 different methods:
echo   1. Naive fine-tuning
echo   2. Standard EWC
echo   3. Linguistic EWC (our proposal)
echo   4. Random EWC (sanity check)
echo.

python train.py

REM Generate visualizations
echo.
echo Generating visualizations...
python visualize.py

REM Show results
echo.
echo ==========================================
echo Experiment Complete!
echo ==========================================
echo.
echo Results saved to .\results\
echo Check the following files:
echo   - comparison.json (numerical results)
echo   - forgetting_comparison.png
echo   - final_accuracy.png
echo   - learning_curves.png
echo.

pause
