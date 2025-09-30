# Scripts Module

## Overview

The `scripts` module contains utility scripts and automation tools for the Real-Time Market Intelligence Platform. These scripts handle system initialization, database setup, data migration, maintenance tasks, and operational utilities.

## Purpose

This directory provides standalone executable scripts that:
- Initialize and configure system components
- Set up databases and data structures
- Perform administrative and maintenance tasks
- Automate common operational workflows
- Support development and testing environments

## Available Scripts

### Database Initialization

#### `initialize_db.py`
Initializes the ClickHouse database with the required schema, tables, and indexes.

**Usage:**
```bash
python src/scripts/initialize_db.py
```

**Features:**
- Creates database schema if it doesn't exist
- Sets up tables for market data, analytics, and metadata
- Creates indexes for query optimization
- Configures partitioning strategies
- Validates database connectivity

**Environment Variables:**
- `CLICKHOUSE_HOST`: Database host (default: localhost)
- `CLICKHOUSE_PORT`: Database port (default: 9000)
- `CLICKHOUSE_USER`: Database user
- `CLICKHOUSE_PASSWORD`: Database password
- `CLICKHOUSE_DATABASE`: Database name (default: market_intelligence)

## Script Categories

### Setup Scripts
Scripts for initial system setup and configuration:
- Database initialization
- Schema creation
- Index generation
- Initial data loading

### Maintenance Scripts
Scripts for ongoing system maintenance:
- Data cleanup
- Archive old data
- Optimize database
- Update indexes

### Migration Scripts
Scripts for data and schema migrations:
- Version upgrades
- Schema changes
- Data transformations
- Backward compatibility

### Utility Scripts
General-purpose utility scripts:
- Data validation
- System health checks
- Performance benchmarking
- Configuration testing

## Development Guidelines

### Creating New Scripts

When creating new scripts:

1. **Naming Convention**: Use descriptive, snake_case names (e.g., `backup_database.py`)
2. **Documentation**: Include docstring with purpose, usage, and requirements
3. **Error Handling**: Implement comprehensive error handling and logging
4. **Configuration**: Use environment variables or config files for settings
5. **Idempotency**: Scripts should be safe to run multiple times
6. **Testing**: Include unit tests for complex logic

### Script Template

```python
#!/usr/bin/env python
"""Script description.

Usage:
    python src/scripts/script_name.py [options]

Requirements:
    - Dependency 1
    - Dependency 2

Environment Variables:
    VAR_NAME: Description
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    try:
        logger.info("Starting script...")
        # Script logic here
        logger.info("Script completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## Running Scripts

### From Command Line

```bash
# Run directly
python src/scripts/script_name.py

# With arguments
python src/scripts/script_name.py --option value

# With environment variables
VAR_NAME=value python src/scripts/script_name.py
```

### From Docker

```bash
# Run script in Docker container
docker-compose exec app python src/scripts/script_name.py
```

### Scheduled Execution

For automated execution, use cron (Linux/Mac) or Task Scheduler (Windows):

```bash
# Example cron entry (daily at 2 AM)
0 2 * * * cd /path/to/project && python src/scripts/maintenance.py
```

## Best Practices

1. **Logging**: Always use logging instead of print statements
2. **Exit Codes**: Return 0 for success, non-zero for failures
3. **Dry Run**: Implement `--dry-run` flag for testing
4. **Confirmation**: Prompt for confirmation on destructive operations
5. **Progress**: Show progress for long-running operations
6. **Rollback**: Implement rollback mechanisms where applicable

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure you're running from the project root
- Check that virtual environment is activated
- Verify all dependencies are installed

**Permission Errors**
- Check file/directory permissions
- Ensure database credentials are correct
- Verify network access to required services

**Configuration Errors**
- Verify all required environment variables are set
- Check configuration file syntax
- Validate connection strings and credentials

## Development Status

This module is currently under active development as part of Phase 2 of the project roadmap.

## Future Enhancements

- Interactive script execution with CLI prompts
- Automated backup and restore utilities
- Performance monitoring and profiling scripts
- Data quality validation tools
- Automated testing and deployment scripts
