@echo off
echo ========================================
echo Full Enterprise Pipeline
echo ========================================

echo.
echo [1/6] Code quality and security checks...
echo Running Black formatter...
black optimizers test_functions visualization tests benchmarks ml_examples
echo Running Flake8 linter...
flake8 optimizers test_functions visualization --max-line-length=88 --ignore=E203,W503
echo Running MyPy type checker...
mypy optimizers test_functions visualization --ignore-missing-imports
echo Running Bandit security scanner...
bandit -r optimizers

echo.
echo [2/6] Running comprehensive test suite...
pytest tests/ --cov=optimizers --cov=test_functions --cov=visualization --cov-report=html --cov-report=xml
if %errorlevel% neq 0 (
    echo WARNING: Some tests failed
)

echo.
echo [3/6] Performance benchmarking...
python benchmarks\scipy_comparison.py
python benchmarks\performance_profiler.py

echo.
echo [4/6] ML examples validation...
python ml_examples\neural_network_optimization.py
python ml_examples\logistic_regression_optimization.py

echo.
echo [5/6] Skipping Docker validation...
echo Docker tests can be run separately with setup_docker.bat

echo.
echo [6/6] Package validation...
python setup.py check
python -m build
twine check dist/*

echo.
echo ========================================
echo Full pipeline completed!
echo.
echo Generated artifacts:
echo - htmlcov\index.html (coverage report)
echo - dist\ (package distributions)
echo - *.png (benchmark plots)
echo ========================================
pause
