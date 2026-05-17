@echo off
REM Quick Dataset Addition Script for Windows
REM Usage: add_dataset.bat <dataset_name> <images_directory> [good_images_directory]

if "%~2"=="" (
    echo Usage: %0 ^<dataset_name^> ^<images_directory^> [good_images_directory]
    echo.
    echo Examples:
    echo   %0 my_dataset C:\path\to\pcb\images
    echo   %0 custom_pcb C:\path\to\defective\pcb\images C:\path\to\good\pcb\images
    echo.
    echo This will:
    echo 1. Create dataset structure in data\^<dataset_name^>\
    echo 2. Split images into train/val/test sets
    echo 3. Copy corresponding label files
    echo 4. Create data.yaml for YOLO training
    goto :eof
)

set DATASET_NAME=%1
set IMAGES_DIR=%2
set GOOD_IMAGES_DIR=%3

echo 🔧 Adding new dataset: %DATASET_NAME%
echo    Images: %IMAGES_DIR%
echo    Good images: %GOOD_IMAGES_DIR%
echo.

REM Run the Python preparation script
if defined GOOD_IMAGES_DIR (
    python prepare_dataset.py --action new --name "%DATASET_NAME%" --images "%IMAGES_DIR%" --good-images "%GOOD_IMAGES_DIR%"
) else (
    python prepare_dataset.py --action new --name "%DATASET_NAME%" --images "%IMAGES_DIR%"
)

echo.
echo ✅ Dataset prepared!
echo.
echo Next steps:
echo 1. Review dataset: data\%DATASET_NAME%\
echo 2. Train model: python train_yolo_simple.py
echo 3. Test: python simple_webcam_server.py