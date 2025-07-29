@echo off
echo ========================================
echo Optimization Toolkit - System Run
echo ========================================

echo.
echo [1/4] Running code quality checks...
echo Running Black formatter...
black optimizers test_functions visualization tests benchmarks
echo Running Flake8 linter...
flake8 optimizers test_functions visualization --max-line-length=88
echo Skipping MyPy type checker (type annotations needed)...

echo.
echo [2/4] Running tests...
pytest tests/ --cov=optimizers --cov=test_functions --cov=visualization --cov-report=html
if %errorlevel% neq 0 (
    echo WARNING: Some tests failed
)

echo.
echo [3/4] Running ML optimization examples...
python ml_examples\neural_network_optimization.py
python ml_examples\logistic_regression_optimization.py

echo.
echo [4/4] Running benchmarks...
if not exist output mkdir output
python benchmarks\scipy_comparison.py
python benchmarks\performance_profiler.py
python example_usage.py

echo.
echo ========================================
echo System run completed!
echo Check htmlcov\index.html for coverage report
if exist output\*.png (
    echo Check output\ folder for generated plots
) else (
    echo No plots generated
)
echo ========================================
pause
