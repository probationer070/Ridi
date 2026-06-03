@echo off
echo Changing directory to script location...
cd /d %~dp0

echo Checking for 'mini' virtual environment...

conda env list | findstr /i "mini" > nul
if %errorlevel% neq 0 (
    echo 'mini' virtual environment not found. Creating...
    call conda env create -f environment.yml
    if %errorlevel% neq 0 (
        echo Error creating 'mini' virtual environment. Please check your environment.yml file and conda installation.
        pause
        exit /b 1
    )
    echo 'mini' virtual environment created successfully.
) else (
    echo 'mini' virtual environment found.
)

echo Activating mini virtual environment...
call conda activate mini

echo Installing Pytorch+cuda related packages...
python install.py
if %errorlevel% neq 0 (
    echo Error installing torch related packages.
    pause
    exit /b 1
)

echo Checking for existing llama_cpp_python installation...
call pip list | findstr /i "llama_cpp_python" > nul
if %errorlevel% equ 0 (
    echo llama-cpp-python found. Skipping installation.
) else (
    echo llama-cpp-python not found. Installing...
    call pip install llama-cpp-python
    if %errorlevel% neq 0 (
        echo Error installing llama-cpp-python.
        pause
        exit /b 1
    )
    echo llama-cpp-python installed.
)

echo Installing requirements.txt...
call pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing requirements.txt.
    pause
    exit /b 1
)
echo Complete to install requirements

@REM echo Check & reinstall ALL requirements
@REM echo Checking installed packages...
@REM pip list > installed_packages.txt
@REM echo Installed packages list saved to installed_packages.txt

echo Installation complete. AII Process finish. Close plz
pause
