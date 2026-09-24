@echo off
REM Activate the conda environment for the cyberbully text detection project.
REM Run this from an Anaconda/Miniconda prompt or a terminal where conda is available.
conda activate cyberbully_text_detection
if errorlevel 1 (
  echo Failed to activate the environment. Make sure conda is installed and accessible.
) else (
  echo Activated conda environment: cyberbully_text_detection
)
