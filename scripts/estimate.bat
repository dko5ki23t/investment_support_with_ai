@echo off

set LEARNING_DIR=

for /f "tokens=1-2" %%i in (path_info.txt) do (
    if %%i == learning_dir (
        set LEARNING_DIR=%CD%\%%j
    )
)

if "%3" == "" (
    echo Usage : estimate.bat term now gain
    exit /b
)

python ..\source\estimate\estimate.py -d %LEARNING_DIR% -t %1 -n %2 -g %3

pause
