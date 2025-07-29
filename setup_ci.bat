@echo off
echo ========================================
echo CI/CD Environment Setup
echo ========================================

echo.
echo [1/4] Checking Git repository...
if not exist .git (
    echo Initializing Git repository...
    git init
    git branch -M main
)

echo.
echo [2/4] Setting up pre-commit hooks...
pre-commit install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install pre-commit hooks
    pause
    exit /b 1
)

echo.
echo [3/4] Validating CI configuration...
if exist .github\workflows\ci.yml (
    echo [OK] CI workflow found
) else (
    echo [MISSING] CI workflow missing
)

if exist docker-compose.yml (
    echo [OK] Docker Compose configuration found
) else (
    echo [MISSING] Docker Compose configuration missing
)

echo.
echo [4/4] Running security checks...
safety check
bandit -r optimizers test_functions visualization

echo.
echo ========================================
echo CI/CD setup completed!
echo.
echo Next steps:
echo 1. git add . ^&^& git commit -m "Initial commit"
echo 2. git remote add origin your-repo-url
echo 3. git push -u origin main
echo ========================================
pause
