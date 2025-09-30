# Tests Module

## Overview

The tests module contains comprehensive test suites for the Real-Time Market Intelligence Platform, ensuring code quality, reliability, and correct functionality across all system components.

## Purpose

This directory provides testing infrastructure for:

- Unit testing of individual components and functions
- Integration testing of module interactions
- End-to-end testing of complete workflows
- Performance and load testing
- API contract testing
- Database testing and data integrity validation

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_api/           # API endpoint tests
│   ├── test_data/          # Data processing tests
│   ├── test_models/        # ML model tests
│   ├── test_streaming/     # Streaming logic tests
│   └── test_utils/         # Utility function tests
├── integration/            # Integration tests
│   ├── test_data_pipeline/ # Data pipeline tests
│   ├── test_api_integration/ # API integration tests
│   └── test_db_operations/ # Database operation tests
├── e2e/                    # End-to-end tests
│   ├── test_user_flows/    # Complete user workflow tests
│   └── test_system/        # System-level tests
├── performance/            # Performance tests
│   ├── test_load/          # Load testing
│   ├── test_stress/        # Stress testing
│   └── test_benchmarks/    # Performance benchmarks
├── fixtures/               # Test data and fixtures
├── mocks/                  # Mock objects and services
├── conftest.py            # Pytest configuration and fixtures
└── README.md              # This file
```

## Testing Framework

### Core Tools
- **pytest**: Main testing framework
- **pytest-cov**: Code coverage reporting
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking and patching
- **pytest-xdist**: Parallel test execution

### Additional Libraries
- **httpx**: HTTP client testing
- **faker**: Test data generation
- **factory_boy**: Test object factories
- **freezegun**: Time mocking
- **responses**: HTTP request mocking

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/unit/test_api/test_endpoints.py

# Run tests matching pattern
pytest -k "test_market_data"
```

### Watch Mode

```bash
# Run tests automatically on file changes
pytest-watch

# With specific path
ptw tests/unit/
```

## Test Categories

### Unit Tests

Test individual components in isolation with mocked dependencies.

```python
# tests/unit/test_data/test_processor.py
import pytest
from src.data.processor import MarketDataProcessor

class TestMarketDataProcessor:
    def test_process_price_data(self):
        processor = MarketDataProcessor()
        raw_data = {"price": "100.50", "volume": "1000"}
        
        result = processor.process(raw_data)
        
        assert result["price"] == 100.50
        assert result["volume"] == 1000
    
    def test_handles_invalid_data(self):
        processor = MarketDataProcessor()
        
        with pytest.raises(ValueError):
            processor.process({"invalid": "data"})
```

### Integration Tests

Test interactions between multiple components.

```python
# tests/integration/test_data_pipeline/test_pipeline.py
import pytest
from src.data import DataIngestion, DataProcessor, DataStorage

@pytest.mark.integration
class TestDataPipeline:
    def test_complete_pipeline(self, test_db):
        # Setup
        ingestion = DataIngestion()
        processor = DataProcessor()
        storage = DataStorage(test_db)
        
        # Execute
        raw_data = ingestion.fetch_market_data("AAPL")
        processed = processor.process(raw_data)
        storage.save(processed)
        
        # Verify
        saved_data = storage.get_latest("AAPL")
        assert saved_data["symbol"] == "AAPL"
        assert saved_data["processed"] is True
```

### End-to-End Tests

Test complete user workflows from start to finish.

```python
# tests/e2e/test_user_flows/test_dashboard.py
import pytest
from playwright.sync_api import Page

@pytest.mark.e2e
class TestDashboardFlow:
    def test_user_views_market_data(self, page: Page):
        # Navigate to dashboard
        page.goto("http://localhost:3000")
        
        # Add asset to watchlist
        page.fill("input[name='symbol']", "AAPL")
        page.click("button:text('Add')")
        
        # Verify data displayed
        assert page.is_visible("text=AAPL")
        assert page.is_visible("text=Price:")
```

### Performance Tests

Test system performance under various loads.

```python
# tests/performance/test_load/test_api_load.py
import pytest
import asyncio
from locust import HttpUser, task, between

class MarketAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def get_market_data(self):
        self.client.get("/api/v1/market/AAPL")
    
    @task(3)
    def get_multiple_assets(self):
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            self.client.get(f"/api/v1/market/{symbol}")
```

## Test Configuration

### conftest.py

Shared fixtures and configuration:

```python
# tests/conftest.py
import pytest
import asyncio
from sqlalchemy import create_engine
from src.config import get_test_config

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def test_db():
    """Create test database."""
    config = get_test_config()
    engine = create_engine(config.database_url)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(engine)

@pytest.fixture
def api_client():
    """Create test API client."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    return TestClient(app)

@pytest.fixture
def mock_kafka_producer(mocker):
    """Mock Kafka producer."""
    return mocker.patch("kafka.KafkaProducer")
```

## Test Data Management

### Fixtures

Create reusable test data:

```python
# tests/fixtures/market_data.py
import pytest
from datetime import datetime

@pytest.fixture
def sample_market_data():
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 1000000,
        "timestamp": datetime.now(),
        "change_percent": 2.5
    }

@pytest.fixture
def multiple_assets():
    return [
        {"symbol": "AAPL", "price": 150.25},
        {"symbol": "MSFT", "price": 280.50},
        {"symbol": "GOOGL", "price": 2800.75}
    ]
```

### Factories

Generate test objects dynamically:

```python
# tests/factories/models.py
import factory
from src.models import MarketData

class MarketDataFactory(factory.Factory):
    class Meta:
        model = MarketData
    
    symbol = factory.Sequence(lambda n: f"SYM{n}")
    price = factory.Faker("pydecimal", left_digits=3, right_digits=2, positive=True)
    volume = factory.Faker("random_int", min=1000, max=10000000)
    timestamp = factory.Faker("date_time_this_year")
```

## Code Coverage

### Coverage Goals
- Overall: >80%
- Critical paths: >95%
- API endpoints: >90%
- Business logic: >90%

### Generate Reports

```bash
# HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=src --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=src --cov-report=xml
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Best Practices

### Test Organization
1. **One test file per module**: Mirror source structure
2. **Descriptive names**: Use clear, action-oriented names
3. **AAA pattern**: Arrange, Act, Assert
4. **Test isolation**: Each test should be independent
5. **Fast execution**: Keep unit tests under 1 second

### Writing Tests
```python
# Good
def test_calculates_moving_average_correctly():
    # Arrange
    data = [1, 2, 3, 4, 5]
    calculator = MovingAverageCalculator(window=3)
    
    # Act
    result = calculator.calculate(data)
    
    # Assert
    assert result == [2.0, 3.0, 4.0]

# Avoid
def test_calculator():
    calc = MovingAverageCalculator(window=3)
    assert calc.calculate([1,2,3,4,5]) == [2,3,4]
```

### Mocking
```python
# Mock external dependencies
def test_api_call(mocker):
    # Mock external API
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"price": 150.25}
    mocker.patch("requests.get", return_value=mock_response)
    
    # Test
    result = get_market_price("AAPL")
    assert result == 150.25
```

## Debugging Tests

```bash
# Run with print statements
pytest -s

# Run with debugger
pytest --pdb

# Stop on first failure
pytest -x

# Verbose output
pytest -vv

# Show local variables on failure
pytest -l
```

## Test Markers

Organize tests with markers:

```python
# Mark slow tests
@pytest.mark.slow
def test_large_dataset_processing():
    pass

# Mark tests requiring database
@pytest.mark.database
def test_data_persistence():
    pass

# Skip tests conditionally
@pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
def test_kafka_integration():
    pass
```

Run specific markers:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m database    # Run only database tests
```

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure `PYTHONPATH` includes project root
- Check `__init__.py` files exist
- Verify dependencies are installed

**Async Test Failures**
- Use `pytest-asyncio` plugin
- Mark async tests with `@pytest.mark.asyncio`
- Ensure event loop is properly configured

**Database Test Issues**
- Check test database connection
- Verify migrations are applied
- Ensure proper cleanup in fixtures

**Flaky Tests**
- Identify race conditions
- Add proper waits/timeouts
- Use freezegun for time-dependent tests

## Contributing

1. Write tests for all new features
2. Maintain minimum coverage standards
3. Follow naming conventions
4. Document complex test scenarios
5. Keep tests fast and focused

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test-Driven Development](https://testdriven.io/)

## Development Status

This module is actively maintained with continuous expansion of test coverage across all platform components.
