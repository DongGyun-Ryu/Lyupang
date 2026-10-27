@echo off
setlocal
if not exist "%LOCALAPPDATA%\Lyupang\data" (
    echo No collected data yet. Run START.cmd first.
    pause
    exit /b 1
)
explorer.exe "%LOCALAPPDATA%\Lyupang\data"
