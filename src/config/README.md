# Configuration Module

## Overview

The `config` module centralizes all configuration management for the Real-Time Market Intelligence Platform. This module provides a unified interface for accessing application settings, environment variables, and configuration parameters across all components.

## Structure

This module contains:

- **Configuration Files**: YAML/JSON configuration templates
- **Settings Classes**: Python classes that encapsulate configuration logic
- **Validators**: Configuration validation and type checking
- **Loaders**: Environment-specific configuration loaders

## Key Responsibilities

### Configuration Management
- Load configuration from multiple sources (files, environment variables, defaults)
- Provide typed access to configuration parameters
- Validate configuration values on application startup
- Support environment-specific configurations (development, staging, production)

### Service Configuration
- **Database Settings**: ClickHouse connection parameters, pool sizes, timeouts
- **Kafka Settings**: Bootstrap servers, topics, consumer groups, producer configs
- **Redis Settings**: Connection strings, cache TTLs, pub/sub channels
- **API Settings**: Rate limits, authentication tokens, endpoint configurations

### Security
- Secure handling of sensitive credentials
- Integration with secret management systems
- Environment variable validation
- Credential rotation support

## Configuration Hierarchy

The configuration system follows a priority order:

1. **Environment Variables**: Highest priority (override all others)
2. **Configuration Files**: Environment-specific YAML/JSON files
3. **Default Values**: Built-in defaults defined in code

## Configuration Structure

```yaml
# Example configuration structure
app:
  name: "Real-Time Market Intelligence Platform"
  environment: "production"
  debug: false
  log_level: "INFO"

database:
  clickhouse:
    host: "localhost"
    port: 9000
    database: "market_intelligence"
    user: "${CLICKHOUSE_USER}"
    password: "${CLICKHOUSE_PASSWORD}"
    pool_size: 10

kafka:
  bootstrap_servers:
    - "kafka1:9092"
    - "kafka2:9092"
  topics:
    market_data: "market-data"
    news_feed: "news-feed"
    sentiment: "sentiment-analysis"
  consumer_group: "market-intelligence-consumers"

redis:
  host: "localhost"
  port: 6379
  password: "${REDIS_PASSWORD}"
  db: 0
  ttl: 3600

api:
  alpha_vantage:
    api_key: "${ALPHA_VANTAGE_KEY}"
    rate_limit: 5  # requests per minute
  yahoo_finance:
    enabled: true
    timeout: 30
```

## Usage Example

```python
from src.config import get_config

# Load configuration
config = get_config()

# Access configuration values
db_host = config.database.clickhouse.host
kafka_servers = config.kafka.bootstrap_servers
api_key = config.api.alpha_vantage.api_key

# Environment-specific configuration
if config.app.environment == "production":
    # Production-specific logic
    pass
```

## Environment Variables

The following environment variables are used:

### Database
- `CLICKHOUSE_HOST`: ClickHouse server hostname
- `CLICKHOUSE_PORT`: ClickHouse server port
- `CLICKHOUSE_USER`: Database username
- `CLICKHOUSE_PASSWORD`: Database password
- `CLICKHOUSE_DATABASE`: Database name

### Kafka
- `KAFKA_BOOTSTRAP_SERVERS`: Comma-separated list of Kafka brokers
- `KAFKA_CONSUMER_GROUP`: Consumer group ID

### Redis
- `REDIS_HOST`: Redis server hostname
- `REDIS_PORT`: Redis server port
- `REDIS_PASSWORD`: Redis password

### API Keys
- `ALPHA_VANTAGE_KEY`: Alpha Vantage API key
- `YAHOO_FINANCE_KEY`: Yahoo Finance API key (if required)
- `TWITTER_API_KEY`: Twitter API key
- `TWITTER_API_SECRET`: Twitter API secret

## Development Status

This module is currently under active development as part of Phase 2 of the project roadmap.

## Best Practices

1. **Never commit sensitive credentials** to version control
2. **Use environment variables** for all sensitive data
3. **Validate configuration** on application startup
4. **Document all configuration options** with examples
5. **Provide sensible defaults** for non-critical settings
6. **Use type hints** for all configuration classes

## Future Enhancements

- Dynamic configuration reloading without restart
- Configuration versioning and audit logging
- Integration with HashiCorp Vault or AWS Secrets Manager
- Configuration validation schemas with detailed error messages
- Web UI for configuration management
