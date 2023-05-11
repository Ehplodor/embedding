# This .bat file checks if the venv directory exists. If it doesn't exist, it creates and initializes the virtual environment using python -m venv venv. Afterward, it proceeds with activating the virtual environment, executing the script, and deactivating the virtual environment as before.
#This updated .bat file ensures that the virtual environment is created and initialized before attempting to activate it and execute the script.

@echo off

REM Set the path to the Python interpreter
set PYTHON_EXECUTABLE=venv\Scripts\python.exe

REM Set the path to the activate_and_run.py script
set SCRIPT_PATH=activate_and_run.py

REM Check if the venv directory exists
if not exist venv (
    REM Create and initialize the virtual environment
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Execute the script
%PYTHON_EXECUTABLE% %SCRIPT_PATH%

REM Deactivate the virtual environment
call venv\Scripts\deactivate.bat
