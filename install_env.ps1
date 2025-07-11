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

# Install PyTorch with CUDA support (CRITICAL: This must complete first)
Write-Host "`nInstalling PyTorch + CUDA (this may take several minutes)..." -ForegroundColor Yellow
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
Check-LastCommand "PyTorch installation"

# Verify PyTorch installation
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
Check-LastCommand "PyTorch verification"

# Install build tools
Write-Host "Installing build tools (cmake, ninja)..." -ForegroundColor Yellow
conda install -y cmake ninja -c nvidia/label/cuda-11.8.0
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
pip install https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118/kaolin-0.17.0-cp311-cp311-win_amd64.whl
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
