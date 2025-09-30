"""Real-Time Market Intelligence Platform.

This package provides a comprehensive platform for real-time financial market data
analysis, processing, and visualization. It integrates multiple data sources and
provides actionable insights through advanced analytics.

Modules:
    api: RESTful API and WebSocket endpoints for data access
    config: Centralized configuration management
    data: Data ingestion, processing, and management
    models: Machine learning models for prediction and analysis
    scripts: Utility scripts and database initialization
    streaming: Real-time stream processing with Kafka
    utils: Utility functions and helpers
    visualization: Data visualization components

Author: Gabriel Afis
License: MIT
"""

__version__ = "0.2.0"
__author__ = "Gabriel Afis"
__license__ = "MIT"

# Expose key components at package level
__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
