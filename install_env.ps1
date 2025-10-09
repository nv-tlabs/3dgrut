# To run this script from PowerShell, navigate to the project folder, and run:
# .\install_env.ps1

param (
    [string]$CondaEnv = "3dgrut"
)

# Function to check if last command succeeded
function Check-LastCommand {
    param($StepName)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: $StepName failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "$StepName completed successfully" -ForegroundColor Green
}

# Function to find Visual Studio cl.exe path
function Find-VisualStudioCompiler {
    Write-Host "Searching for Visual Studio C++ compiler..." -ForegroundColor Yellow
    
    # Search paths for different VS editions and versions
    $searchPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\*\bin\Hostx64\x64", 
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
        # Fallback to x86 host if x64 host not available
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x86",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\*\bin\Hostx64\x86", 
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x86",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x86",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x86",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\*\bin\Hostx64\x86", 
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x86",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x86"
    )
    
    foreach ($path in $searchPaths) {
        $resolvedPaths = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        foreach ($resolvedPath in $resolvedPaths) {
            $clExe = Join-Path $resolvedPath.FullName "cl.exe"
            if (Test-Path $clExe) {
                Write-Host "Found Visual Studio compiler at: $($resolvedPath.FullName)" -ForegroundColor Green
                return $resolvedPath.FullName
            }
        }
    }
    
    Write-Host "Warning: Could not find Visual Studio C++ compiler automatically." -ForegroundColor Yellow
    Write-Host "You may need to install Visual Studio Build Tools or add cl.exe to PATH manually." -ForegroundColor Yellow
    return $null
}

Write-Host "`nStarting Conda environment setup: $CondaEnv"

# Initialize conda for PowerShell (this enables conda commands)
Write-Host "Initializing conda for PowerShell..."
& conda init powershell
Check-LastCommand "Conda initialization"

# Refresh the current session to pick up conda changes
Write-Host "Refreshing PowerShell session..."
& powershell -Command "& {conda --version}"
Check-LastCommand "Conda verification"

Write-Host "Creating conda environment..."
conda create -n $CondaEnv python=3.11 -y
Check-LastCommand "Conda environment creation"

Write-Host "Activating conda environment..."
conda activate $CondaEnv
Check-LastCommand "Conda environment activation"

# Verify environment is active
Write-Host "Verifying environment activation..."
$CurrentEnv = $env:CONDA_DEFAULT_ENV
if ($CurrentEnv -ne $CondaEnv) {
    Write-Host "Warning: Expected environment '$CondaEnv' but found '$CurrentEnv'" -ForegroundColor Yellow
}
Write-Host "Current environment: $CurrentEnv" -ForegroundColor Green

# Configure Visual Studio C++ compiler for PyTorch JIT compilation
Write-Host "`nConfiguring Visual Studio C++ compiler for conda environment..." -ForegroundColor Yellow
$vsCompilerPath = Find-VisualStudioCompiler
if ($vsCompilerPath) {
    # Get current PATH from conda environment
    $currentPath = conda env config vars list -n $CondaEnv | Select-String "PATH" | ForEach-Object { $_.ToString().Split('=', 2)[1] }
    
    if ($currentPath) {
        # Append to existing PATH
        $newPath = "$vsCompilerPath;$currentPath"
    } else {
        # Create new PATH with VS compiler and current system PATH
        $newPath = "$vsCompilerPath;$env:PATH"
    }
    
    # Set the PATH environment variable for this conda environment
    conda env config vars set -n $CondaEnv "PATH=$newPath"
    Check-LastCommand "Visual Studio compiler PATH configuration"
    
    Write-Host "Visual Studio compiler added to conda environment PATH" -ForegroundColor Green
    Write-Host "The compiler will be automatically available when you activate the environment" -ForegroundColor Green
    
    # Reactivate environment to pick up the new PATH
    Write-Host "Reactivating environment to apply PATH changes..." -ForegroundColor Yellow
    conda deactivate
    conda activate $CondaEnv
    Check-LastCommand "Environment reactivation"
    
    # Verify the compiler is now accessible
    Write-Host "Verifying compiler accessibility..." -ForegroundColor Yellow
    $clTest = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($clTest) {
        Write-Host "Visual Studio compiler (cl.exe) is now accessible: $($clTest.Source)" -ForegroundColor Green
    } else {
        Write-Host "Warning: cl.exe still not found in PATH. Manual setup may be required." -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping compiler PATH setup - Visual Studio not found" -ForegroundColor Yellow
    Write-Host "You may need to install Visual Studio Build Tools and re-run this script" -ForegroundColor Yellow
}

# Install CUDA toolkit 12.4 (nvcc, headers, libs)
Write-Host "`nInstalling CUDA toolkit 12.4 (nvcc, etc.)..." -ForegroundColor Yellow
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
Check-LastCommand "CUDA toolkit installation"

# Install PyTorch with CUDA support (CRITICAL: This must complete first)
Write-Host "`nInstalling PyTorch + CUDA 12.4 (this may take several minutes)..." -ForegroundColor Yellow
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
Check-LastCommand "PyTorch installation"

# Verify PyTorch installation
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
Check-LastCommand "PyTorch verification"

# Install build tools
Write-Host "Installing build tools (cmake, ninja)..." -ForegroundColor Yellow
conda install -y cmake ninja -c nvidia/label/cuda-12.4.0
Check-LastCommand "Build tools installation"

# Initialize Git submodules
Write-Host "Initializing Git submodules..." -ForegroundColor Yellow
git submodule update --init --recursive
Check-LastCommand "Git submodules initialization"

# Install Python dependencies
Write-Host "Installing Python requirements from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
Check-LastCommand "Requirements installation"

# Install additional dependencies
Write-Host "Installing Cython..." -ForegroundColor Yellow
pip install cython
Check-LastCommand "Cython installation"

Write-Host "Installing Hydra-core..." -ForegroundColor Yellow
pip install hydra-core
Check-LastCommand "Hydra-core installation"

# Install Kaolin
Write-Host "Installing Kaolin (this may take a while)..." -ForegroundColor Yellow
pip install https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124/kaolin-0.17.0-cp311-cp311-win_amd64.whl
Check-LastCommand "Kaolin installation"

# Install project in development mode
Write-Host "Installing project in development mode..." -ForegroundColor Yellow
pip install -e .
Check-LastCommand "Project installation"

# Final success message
Write-Host "`n" -NoNewline
Write-Host "=================================================" -ForegroundColor Green
Write-Host "    INSTALLATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host "Environment '$CondaEnv' is ready with all dependencies!" -ForegroundColor Green
Write-Host ""
Write-Host "To use the environment:" -ForegroundColor Cyan
Write-Host "  conda activate $CondaEnv" -ForegroundColor White
