@echo off
setlocal
rem ============================================================
rem  Matched-subset ablation eval - Windows host runner.
rem  10k stratified subset, 4-crop (multi) mode, 11 arms:
rem    base mono_K1 ep002 + unfrozen/frozen 1-epoch ablations.
rem
rem  RESUMABLE: arms whose metrics JSON already exists are skipped,
rem  so the script can be re-run after a crash / interruption.
rem
rem  Prerequisites:
rem    - scripts\setup_deps.bat has been run on the host (creates
rem      .venv with requirements.txt + ..\bop_toolkit).
rem    - stereo_checkpoints\ holds the checkpoints (mono_K1\epoch_002
rem      and ablation\chkpt_stereobj-1m_*\epoch_000).
rem
rem  Overrides - set these in the shell before running, e.g.:
rem    set DATA_ROOT=E:\data
rem    set N=2500
rem  then: scripts\eval_stereobj_ablation_10k.bat
rem ============================================================

set "REPO=%~dp0.."
if not defined PYTHON      set "PYTHON=%REPO%\.venv\Scripts\python.exe"
if not defined CKPT_ROOT   set "CKPT_ROOT=%REPO%\stereo_checkpoints"
if not defined EVAL_ROOT   set "EVAL_ROOT=%REPO%\reports\ablation_10k_multi"
if not defined DATA_ROOT   set "DATA_ROOT=D:/6DPose"
if not defined N           set "N=10000"
if not defined SUFFIX      set "SUFFIX=multi"
if not defined CROP        set "CROP=multi"
if not defined NUM_WORKERS set "NUM_WORKERS=2"

if not exist "%PYTHON%" (
    echo ERROR: python not found: %PYTHON%
    echo        run scripts\setup_deps.bat first to create the venv.
    exit /b 1
)
if not exist "%CKPT_ROOT%" (
    echo ERROR: checkpoint root not found: %CKPT_ROOT%
    exit /b 1
)
if not exist "%EVAL_ROOT%" mkdir "%EVAL_ROOT%"

rem The dataset code resolves ^<DATA_ROOT^>\stereobj-1m, but the host folder
rem is named SteroOBJ-1M. Bridge with a junction (no admin rights needed).
set "DATA_BS=%DATA_ROOT:/=\%"
if exist "%DATA_BS%\stereobj-1m" goto have_data
if exist "%DATA_BS%\SteroOBJ-1M" (
    mklink /J "%DATA_BS%\stereobj-1m" "%DATA_BS%\SteroOBJ-1M"
) else (
    echo ERROR: neither %DATA_BS%\stereobj-1m nor %DATA_BS%\SteroOBJ-1M exists
    echo        ^(override with set DATA_ROOT=...^)
    exit /b 1
)
:have_data

rem CROP=multi keeps the 4-crop confidence argmax; anything else runs
rem --single_crop (the miscalibrated-argmax caveat, see reports/ history).
set "CROP_FLAGS=--single_crop"
if /i "%CROP%"=="multi" set "CROP_FLAGS="

call :run_arm "chkpt_stereobj-1m_mono_K1"    "epoch_002" "mono"   "mono_K1"
call :run_arm "ablation/chkpt_stereobj-1m_uf_left_k1"    "epoch_000" "stereo" "uf_left_k1"
call :run_arm "ablation/chkpt_stereobj-1m_uf_mean_k1"    "epoch_000" "stereo" "uf_mean_k1"
call :run_arm "ablation/chkpt_stereobj-1m_uf_reg_k1"     "epoch_000" "stereo" "uf_reg_k1"
call :run_arm "ablation/chkpt_stereobj-1m_uf_regclf_k1"  "epoch_000" "stereo" "uf_regclf_k1"
call :run_arm "ablation/chkpt_stereobj-1m_uf_regcorr_k1" "epoch_000" "stereo" "uf_regcorr_k1"
call :run_arm "ablation/chkpt_stereobj-1m_uf_full_k1"    "epoch_000" "stereo" "uf_full_k1"
call :run_arm "ablation/chkpt_stereobj-1m_fr_full_k1"    "epoch_000" "stereo" "fr_full_k1"
call :run_arm "ablation/chkpt_stereobj-1m_fr_reg_k1"     "epoch_000" "stereo" "fr_reg_k1"
call :run_arm "ablation/chkpt_stereobj-1m_fr_regclf_k1"  "epoch_000" "stereo" "fr_regclf_k1"
call :run_arm "ablation/chkpt_stereobj-1m_fr_regcorr_k1" "epoch_000" "stereo" "fr_regcorr_k1"

echo [%TIME%] Building summary tables...
"%PYTHON%" "%REPO%\tools\ablation_table.py" ^
    --reports_dir "%EVAL_ROOT%" ^
    --dataset stereobj-1m --split val ^
    --suffix "n%N%_strat_%SUFFIX%" ^
    --dataset_root "%DATA_ROOT%" ^
    --output "%EVAL_ROOT%\summary_tables.md"

echo [%TIME%] ALL DONE - results in %EVAL_ROOT%
endlocal
exit /b 0

:run_arm
set "SUBDIR=%~1"
set "MODEL=%~2"
set "MODE=%~3"
set "TAG=%~4_n%N%_strat_%SUFFIX%"
set "CSVNAME=mrcnet_stereobj-1m-val_%SUBDIR:/=_%_%TAG%"
if exist "%EVAL_ROOT%\%TAG%.json" (
    echo [%TIME%] SKIP %TAG% ^(already scored^)
    exit /b 0
)
if not exist "%CKPT_ROOT%\%SUBDIR:\=/%\%MODEL%.pth" (
    echo [%TIME%] MISSING checkpoint %CKPT_ROOT%\%SUBDIR%\%MODEL%.pth
    exit /b 0
)
echo [%TIME%] START %SUBDIR% / %MODEL% (%MODE%, n=%N%, crop=%CROP%)
pushd "%CKPT_ROOT%"
"%PYTHON%" "%REPO%\inference_stereobj.py" --dataset stereobj-1m --split val ^
    --checkpoint_name "%SUBDIR%" --model_name "%MODEL%" ^
    --output_suffix "%TAG%" --stereo_mode "%MODE%" ^
    --max_instances %N% --stratified --no_compile %CROP_FLAGS% ^
    --num_workers %NUM_WORKERS% ^
    --dataset_root "%DATA_ROOT%" --eval_root "%EVAL_ROOT%"
if errorlevel 1 (
    echo [%TIME%] FAILED inference %SUBDIR%/%MODEL%
    popd
    exit /b 0
)
popd
"%PYTHON%" "%REPO%\tools\eval_stereobj.py" ^
    --results "%EVAL_ROOT%\%CSVNAME%.csv" ^
    --dataset stereobj-1m --split val --stereo_mode "%MODE%" ^
    --output "%EVAL_ROOT%\%TAG%.json"
if errorlevel 1 (
    echo [%TIME%] FAILED scoring %TAG%
    exit /b 0
)
echo [%TIME%] OK %TAG%
exit /b 0
