@echo off
echo ========================================
echo Git Repository Setup
echo ========================================

echo.
echo [1/6] Initializing Git repository...
git init
git branch -M main

echo.
echo [2/6] Adding all files...
git add .

echo.
echo [3/6] Creating initial commit...
git commit -m "feat: initial commit with optimization toolkit - core optimizers, test functions, visualization, CI/CD, Docker, ML examples"

echo.
echo [4/6] Setting up pre-commit hooks...
pre-commit install

echo.
echo [5/6] Repository status...
git status
git log --oneline -5

echo.
echo [6/6] Next steps...
echo.
echo To connect to GitHub:
echo 1. Create repository on GitHub
echo 2. git remote add origin https://github.com/NgangaKamau3/optimisation_toolkit.git
echo 3. git push -u origin main
echo.
echo To make changes:
echo 1. git checkout -b feature/your-feature
echo 2. Make changes
echo 3. git add . && git commit -m "feat: your feature"
echo 4. git push origin feature/your-feature
echo 5. Create pull request on GitHub

echo.
echo ========================================
echo Git setup completed!
echo Repository ready for GitHub deployment
echo ========================================
pause
