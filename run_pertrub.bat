@echo off
setlocal enabledelayedexpansion

REM Check if environment name is provided
if "%1"=="" (
    echo Please provide an environment name as the first argument.
    exit /b 1
)

REM Check if number of agents is provided
if "%2"=="" (
    echo Please provide the number of agents as the second argument.
    exit /b 1
)

if "%3"=="" (
    echo Please provide the total rounds for server epoch as the third argument.
    exit /b 1
)

if "%4"=="" (
    echo Please decide use regular policy in KL divergence. 1 for True and 0 for False.
    exit /b 1
)

set env=%1
set agents=%2
set rounds=%3
set use_regul=%4

set "model=RNPPO"
:: Get current date and time with wmic command to ensure consistent format
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"

:: Extract date and time components
set "year=%dt:~0,4%"
set "month=%dt:~4,2%"
set "day=%dt:~6,2%"
set "hour=%dt:~8,2%"
set "minute=%dt:~10,2%"

:: Create time string in format YYYY_MM_DD_HH_MM
set "time_str=%year%_%month%_%day%_%hour%_%minute%"

set "save_dir=multiagent/!time_str!_c!agents!_!env!_!model!_regul!use_regul!"

echo Starting server
start /B python server.py -r %rounds% -c %agents% --log_dir "!save_dir!"

timeout 3 > nul

echo %env%

:: Prepare client command based on use_regul parameter
if "%use_regul%"=="1" (
    set "regul_flag=--kld_use_regul"
) else (
    set "regul_flag="
)

set /a last_agent=%agents% - 1
for /l %%i in (0,1,%last_agent%) do (
    echo Starting client %%i
    start /B python client.py -i %%i -e %env% --log_dir !save_dir! !regul_flag! --model !model!
)

:loop
tasklist /FI "IMAGENAME eq python.exe" |find "python.exe" >nul
if not errorlevel 1 goto loop

echo All processes completed.
pause