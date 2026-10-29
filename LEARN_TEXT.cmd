@echo off
setlocal
rem Correct the kwcli 1.0.3 Windows account-name ambiguity only in this process.
for /f "delims=" %%U in ('whoami') do set "USERNAME=%%U"
set "LYUPANG_KWCLI_PY=%APPDATA%\uv\data\tools\kwcli\Scripts\python.exe"
if not exist "%LYUPANG_KWCLI_PY%" set "LYUPANG_KWCLI_PY=%APPDATA%\uv\tools\kwcli\Scripts\python.exe"
if defined UV_TOOL_DIR set "LYUPANG_KWCLI_PY=%UV_TOOL_DIR%\kwcli\Scripts\python.exe"
if defined LYUPANG_PYTHON set "LYUPANG_KWCLI_PY=%LYUPANG_PYTHON%"
if not exist "%LYUPANG_KWCLI_PY%" (
    echo Cannot locate the Python environment used by kwcli.
    echo Expected: "%LYUPANG_KWCLI_PY%"
    echo Please share this message. Do not enter API keys here.
    pause
    exit /b 1
)
"%LYUPANG_KWCLI_PY%" -X utf8 "%~dp0learn_text.py" %*
set "LYUPANG_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %LYUPANG_RESULT%
