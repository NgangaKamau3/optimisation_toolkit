# Deployment Guide

## Quick Start

### 1. Environment Setup
```bash
# Windows
setup_environment.bat

# Linux/Mac
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

### 2. Run Full System
```bash
# Windows
run_system.bat

# Linux/Mac
make all-checks
python example_usage.py
```

### 3. Deploy to GitHub
```bash
# Windows
deploy_to_github.bat

# Linux/Mac
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/NgangaKamau3/optimisation_toolkit.git
git push -u origin main
```

## CI/CD Configuration

### Required GitHub Secrets
- `PYPI_API_TOKEN`: For PyPI package deployment
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token
- `CODECOV_TOKEN`: Code coverage reporting

### Branch Protection Rules
1. Require pull request reviews
2. Require status checks to pass
3. Require branches to be up to date
4. Include administrators

## Docker Deployment

### Development
```bash
docker-compose up
```

### Production
```bash
docker build -t optimization-toolkit .
docker run -p 8888:8888 optimization-toolkit
```

## Package Distribution

### PyPI Release
1. Update version in `setup.py`
2. Create git tag: `git tag v1.0.0`
3. Push tag: `git push origin v1.0.0`
4. GitHub Actions will automatically deploy

### Manual PyPI Upload
```bash
python -m build
twine upload dist/*
```

## Monitoring & Maintenance

### Code Quality Checks
```bash
make format      # Black formatting
make lint        # Flake8 linting
make type-check  # MyPy type checking
make test        # Run tests with coverage
```

### Performance Monitoring
```bash
python benchmarks/performance_profiler.py
pytest tests/test_performance.py --benchmark-only
```

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure package is installed with `pip install -e .`
2. **Test failures**: Check Python version compatibility (3.8+)
3. **Docker build fails**: Verify Docker daemon is running
4. **CI/CD failures**: Check GitHub secrets are properly configured

### Getting Help
- Check [Issues](https://github.com/NgangaKamau3/optimisation_toolkit/issues)
- Review [Contributing Guidelines](CONTRIBUTING.md)
- Contact maintainers via email
