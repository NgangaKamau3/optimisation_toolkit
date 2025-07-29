@echo off
echo ========================================
echo Fixing CI Issues
echo ========================================

echo.
echo [1/3] Running Black formatter...
black benchmarks ml_examples tests example_usage.py

echo.
echo [2/3] Removing unused imports...
echo Fixing import issues...

echo.
echo [3/3] Committing fixes...
git add .
git commit -m "fix: resolve linting issues for CI/CD pipeline"
git push origin main

echo.
echo ========================================
echo CI fixes pushed to GitHub
echo ========================================
pause
