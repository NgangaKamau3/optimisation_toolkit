@echo off
echo ========================================
echo GitHub Deployment Setup
echo ========================================

echo.
echo [1/5] Initializing Git repository...
git init
git add .
git commit -m "Initial commit: Complete optimization toolkit with ML examples"

echo.
echo [2/5] Adding GitHub remote...
git remote add origin https://github.com/NgangaKamau3/optimisation_toolkit.git

echo.
echo [3/5] Creating main branch...
git branch -M main

echo.
echo [4/5] Setting up pre-commit hooks...
pip install pre-commit
pre-commit install

echo.
echo [5/5] Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo Deployment completed!
echo.
echo Next steps:
echo 1. Go to GitHub repository settings
echo 2. Add secrets for CI/CD:
echo    - PYPI_API_TOKEN (for PyPI deployment)
echo    - DOCKERHUB_USERNAME and DOCKERHUB_TOKEN
echo 3. Enable GitHub Pages for documentation
echo 4. Set up branch protection rules
echo ========================================
pause
