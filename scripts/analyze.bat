@echo off

set FETCHED_DIR=
set ESTIMATE_DIR=

for /f "tokens=1-2" %%i in (path_info.txt) do (
    if %%i == fetched_dir (
        set FETCHED_DIR=%CD%\%%j
    )
    if %%i == estimate_dir (
        set ESTIMATE_DIR=%CD%\%%j
    )
)

python ..\source\analyze\analyze.py -f %FETCHED_DIR% -a %ESTIMATE_DIR%

pause
