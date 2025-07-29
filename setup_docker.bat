@echo off
echo ========================================
echo Docker Environment Setup
echo ========================================

echo.
echo [1/4] Building Docker image...
docker build -t optimization-toolkit .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build Docker image
    pause
    exit /b 1
)

echo.
echo [2/4] Creating Docker network...
docker network create opt-network 2>nul

echo.
echo [3/4] Running tests in container...
docker run --rm -v %cd%:/workspace optimization-toolkit pytest tests/

echo.
echo [4/4] Starting development container...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Docker Compose
    pause
    exit /b 1
)

echo.
echo ========================================
echo Docker setup completed!
echo.
echo Access Jupyter at: http://localhost:8888
echo Access container: docker exec -it optimization-toolkit bash
echo Stop services: docker-compose down
echo ========================================
pause
