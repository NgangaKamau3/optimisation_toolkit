# Contributing to Optimization Toolkit

We welcome contributions! This guide will help you get started.

## Development Setup

### Using Docker (Recommended)
```bash
git clone https://github.com/NgangaKamau3/optimisation_toolkit.git
cd optimisation_toolkit
docker-compose up
```

### Local Development
```bash
git clone https://github.com/NgangaKamau3/optimisation_toolkit.git
cd optimisation_toolkit
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pre-commit install
```

## Code Standards

### Code Quality
- **Black** for formatting: `black .`
- **Flake8** for linting: `flake8 optimizers test_functions visualization`
- **MyPy** for type checking: `mypy optimizers test_functions visualization`
- **Pre-commit** hooks enforce standards automatically

### Testing
- Write tests for all new functionality
- Maintain >90% code coverage
- Run tests: `pytest tests/ --cov`
- Benchmark performance: `pytest tests/ --benchmark-only`

### Documentation
- Add docstrings to all public functions/classes
- Update README.md for new features
- Include mathematical formulations where relevant

## Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/new-optimizer`
3. **Implement** your changes with tests
4. **Run** quality checks: `pre-commit run --all-files`
5. **Submit** a pull request with clear description

## Adding New Optimizers

1. Inherit from `BaseOptimizer`
2. Implement `step()` method
3. Add comprehensive tests
4. Include mathematical documentation
5. Update benchmarks and examples

Example:
```python
class NewOptimizer(BaseOptimizer):
    def step(self, params, grad):
        # Implementation here
        return updated_params
```

## Adding Test Functions

1. Inherit from `BaseFunction`
2. Implement `evaluate()` and `gradient()` methods
3. Define `global_minimum` and `bounds` attributes
4. Add gradient verification tests

## Performance Guidelines

- Profile new code with `memory_profiler` and `line_profiler`
- Benchmark against existing implementations
- Optimize NumPy operations for vectorization
- Document computational complexity

## Issue Reporting

- Use issue templates
- Include minimal reproducible examples
- Specify environment details
- Tag appropriately (bug, enhancement, question)

## Code Review Process

- All changes require review
- Automated checks must pass
- Maintain backward compatibility
- Consider performance implications

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. GitHub Actions handles PyPI deployment

## Community Guidelines

- Be respectful and inclusive
- Help newcomers learn
- Share mathematical insights
- Collaborate on improvements

Thank you for contributing to the optimization community!
