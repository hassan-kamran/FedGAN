# FedGAN Test Suite

This directory contains the test suite for the FedGAN project.

## Structure

```
tests/
├── unit/              # Unit tests for isolated components
│   ├── test_config.py       # Configuration tests
│   ├── test_data.py         # Data loading tests
│   └── test_models.py       # Model architecture tests
├── integration/       # Integration tests
│   └── test_training.py     # Training pipeline tests
├── e2e/              # End-to-end tests
├── fixtures/         # Test fixtures and utilities
└── conftest.py       # Pytest configuration and shared fixtures

```

## Running Tests

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run with coverage
make coverage

# Run specific test file
pytest tests/unit/test_config.py -v

# Run specific test
pytest tests/unit/test_config.py::TestModelConfig::test_default_values -v
```

## Test Markers

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests requiring multiple components
- `@pytest.mark.e2e`: End-to-end tests running full experiments
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.gpu`: Tests requiring GPU

## Coverage

Target: >80% code coverage

View coverage report:
```bash
make coverage
open htmlcov/index.html
```
