@echo off

set FETCHED_DIR=
set STOCKCODE_CSV=

for /f "tokens=1-2" %%i in (path_info.txt) do (
    if %%i == fetched_dir (
        set FETCHED_DIR=%CD%\%%j
    )
    if %%i == stockcode_csv (
        set STOCKCODE_CSV=%CD%\%%j
    )
)

python ..\source\fetch_data\fetch_data.py %STOCKCODE_CSV% %FETCHED_DIR% 

pause
