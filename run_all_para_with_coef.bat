@echo off
setlocal enabledelayedexpansion

:: Setting environment variables for lists
set "clients_list=3"
:: set "environments=CartPoleSwingUpFixInitState-v2 Pendulum-v1 HopperFixLength-v0 HalfCheetahFixLength-v0"
set "environments=HopperFixLength-v0"
set "model_list=PBPPO RNPPO PPO"
set "total_cpu=9"

:: Get current date and time - using PowerShell for better compatibility
for /f "tokens=*" %%a in ('powershell -Command "Get-Date -Format 'yyyy_MM_dd_HH_mm'"') do set "time_str=%%a"

:: Create multiagent directory if it doesn't exist
if not exist "multiagent" (
    echo Creating multiagent directory...
    mkdir "multiagent"
)

set RAY_DEDUP_LOGS=0

:: Set number of repetitions for each experiment
set "num_repetitions=5"

:: Loop through repetitions
for /l %%n in (1,1,%num_repetitions%) do (
    echo.
    echo ========================================
    echo Starting Repetition %%n of %num_repetitions%
    echo ========================================
    echo.
    
    :: Loop through clients
    for %%c in (%clients_list%) do (
        set /a "cpu_per_client=!total_cpu! / %%c"
        if !cpu_per_client! LSS 1 (
            set "cpu_per_client=1"
        )
        echo Debug: Clients: %%c, CPU per client: !cpu_per_client!
        
        :: Loop through environments and their rounds
        for %%e in (%environments%) do (
            
            for %%m in (%model_list%) do (
                
                :: Check if model is PBPPO to determine if we need two runs
                if "%%m"=="PBPPO" (
                    set "regul_list=1 0"
                ) else (
                    set "regul_list=none"
                )
                
                :: Loop through regul variations (only applies to PBPPO)
                for %%r in (!regul_list!) do (
                    
                    :: Generate save directory path with regul suffix for PBPPO
                    :: Add repetition number to directory name
                    if "%%m"=="PBPPO" (
                        set "save_dir=multiagent/!time_str!_%%e_c%%c/%%m_regul%%r/rep%%n"
                        if "%%r"=="1" (
                            set "regul_flag=--kld_use_regul"
                        ) else (
                            set "regul_flag="
                        )
                    ) else (
                        set "save_dir=multiagent/!time_str!_%%e_c%%c/%%m/rep%%n"
                        set "regul_flag="
                    )
                    
                    echo Running experiment with:
                    echo Repetition: %%n of %num_repetitions%
                    echo Environment: %%e
                    echo Rounds: !rounds!
                    echo Clients: %%c
                    echo Model: %%m
                    if "%%m"=="PBPPO" echo Regul: %%r
                    echo Save Directory: !save_dir!
                    echo.
                    
                    :: Start server in a new PowerShell window
                    echo Starting server
                    start /B python server.py -c %%c --log_dir "!save_dir!" -e %%e
                    echo Server started, waiting for 5 seconds...
                    timeout 5 > nul
                    echo Starting clients...
                    set /a last_agent=%%c - 1
                    
                    :: Start all clients in separate PowerShell windows
                    for /l %%i in (0,1,!last_agent!) do (
                        echo Starting client %%i
                        :: /c = close; /k = keep
                        start "Client %%i" cmd /c python client.py -i %%i -e %%e --log_dir "!save_dir!" --n_cpu !cpu_per_client! --model %%m !regul_flag!
                    )
                    
                    :: Start checking script and wait for it to complete
                    echo Waiting for server to complete...
                    start /wait powershell -NoExit -Command "conda activate Fed ; python check_server_end.py --log_dir '!save_dir!' ; exit"
                    
                    echo Current experiment completed.
                    echo.
                    echo ------------------------
                    echo.
                    
                    :: Add a delay between experiments
                    timeout /t 5 /nobreak > nul
                )
            )
        )
    )
)

:: Clean up any remaining marker file
if exist experiment_running.txt del experiment_running.txt

echo All experiments completed.
pause