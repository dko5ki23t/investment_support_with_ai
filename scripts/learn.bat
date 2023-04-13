@echo off

set FETCHED_DIR=
set LEARNING_DIR=

for /f "tokens=1-2" %%i in (path_info.txt) do (
    if %%i == fetched_dir (
        set FETCHED_DIR=%CD%\%%j
    )
    if %%i == learning_dir (
        set LEARNING_DIR=%CD%\%%j
    )
)

python ..\source\learn\learn.py %FETCHED_DIR% %LEARNING_DIR%

pause
