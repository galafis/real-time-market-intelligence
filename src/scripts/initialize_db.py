#!/usr/bin/env python3
"""
ClickHouse Database Initialization Script

This script initializes the ClickHouse database for the Real-Time Market Intelligence Platform.
It creates the necessary databases, tables, and materialized views for storing and querying
financial market data.

Usage:
    python src/scripts/initialize_db.py
"""

import os
import sys
from datetime import datetime
import clickhouse_connect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_clickhouse_client():
    """Establish connection to ClickHouse server."""
    host = os.getenv('CLICKHOUSE_HOST', 'localhost')
    port = int(os.getenv('CLICKHOUSE_PORT', '8123'))
    username = os.getenv('CLICKHOUSE_USER', 'default')
    password = os.getenv('CLICKHOUSE_PASSWORD', '')
    
    try:
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password
        )
        print(f"✓ Connected to ClickHouse at {host}:{port}")
        return client
    except Exception as e:
        print(f"✗ Failed to connect to ClickHouse: {e}")
        sys.exit(1)


def create_database(client):
    """Create the main database for market intelligence."""
    db_name = os.getenv('CLICKHOUSE_DB', 'market_intelligence')
    
    try:
        client.command(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        print(f"✓ Database '{db_name}' created/verified")
        return db_name
    except Exception as e:
        print(f"✗ Failed to create database: {e}")
        sys.exit(1)


def create_tables(client, db_name):
    """Create all required tables for the platform."""
    
    tables = [
        # Market data table
        f"""
        CREATE TABLE IF NOT EXISTS {db_name}.market_data (
            symbol String,
            timestamp DateTime64(3),
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume UInt64,
            source String,
            ingestion_time DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (symbol, timestamp)
        """,
        
        # Sentiment analysis table
        f"""
        CREATE TABLE IF NOT EXISTS {db_name}.sentiment_data (
            symbol String,
            timestamp DateTime,
            source String,
            content String,
            sentiment_score Float32,
            sentiment_label String,
            confidence Float32,
            ingestion_time DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (symbol, timestamp)
        """,
        
        # Predictions table
        f"""
        CREATE TABLE IF NOT EXISTS {db_name}.predictions (
            symbol String,
            prediction_time DateTime,
            target_time DateTime,
            predicted_price Float64,
            confidence_lower Float64,
            confidence_upper Float64,
            model_name String,
            model_version String,
            ingestion_time DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(prediction_time)
        ORDER BY (symbol, prediction_time, target_time)
        """,
        
        # Alerts table
        f"""
        CREATE TABLE IF NOT EXISTS {db_name}.alerts (
            alert_id UUID DEFAULT generateUUIDv4(),
            symbol String,
            alert_type String,
            trigger_condition String,
            trigger_value Float64,
            current_value Float64,
            triggered_at DateTime,
            severity String,
            message String,
            acknowledged Boolean DEFAULT false
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(triggered_at)
        ORDER BY (triggered_at, symbol)
        """,
    ]
    
    for i, table_query in enumerate(tables, 1):
        try:
            client.command(table_query)
            print(f"✓ Table {i}/{len(tables)} created/verified")
        except Exception as e:
            print(f"✗ Failed to create table {i}: {e}")
            sys.exit(1)


def create_materialized_views(client, db_name):
    """Create materialized views for aggregated data."""
    
    views = [
        # Hourly aggregated market data
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {db_name}.market_data_hourly
        ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(hour)
        ORDER BY (symbol, hour)
        AS SELECT
            symbol,
            toStartOfHour(timestamp) AS hour,
            argMin(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            argMax(close, timestamp) AS close,
            sum(volume) AS volume,
            count() AS tick_count
        FROM {db_name}.market_data
        GROUP BY symbol, hour
        """,
        
        # Daily sentiment scores
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {db_name}.sentiment_daily
        ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(day)
        ORDER BY (symbol, day)
        AS SELECT
            symbol,
            toDate(timestamp) AS day,
            avg(sentiment_score) AS avg_sentiment,
            count() AS sentiment_count,
            countIf(sentiment_label = 'positive') AS positive_count,
            countIf(sentiment_label = 'negative') AS negative_count,
            countIf(sentiment_label = 'neutral') AS neutral_count
        FROM {db_name}.sentiment_data
        GROUP BY symbol, day
        """,
    ]
    
    for i, view_query in enumerate(views, 1):
        try:
            client.command(view_query)
            print(f"✓ Materialized view {i}/{len(views)} created/verified")
        except Exception as e:
            print(f"✗ Failed to create materialized view {i}: {e}")
            sys.exit(1)


def verify_setup(client, db_name):
    """Verify the database setup."""
    try:
        # Check tables
        tables = client.query(f"SHOW TABLES FROM {db_name}").result_rows
        print(f"\n✓ Database setup complete. Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        return True
    except Exception as e:
        print(f"✗ Failed to verify setup: {e}")
        return False


def main():
    """Main initialization routine."""
    print("="*60)
    print("ClickHouse Database Initialization")
    print("Real-Time Market Intelligence Platform")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print()
    
    # Connect to ClickHouse
    client = get_clickhouse_client()
    
    # Create database
    db_name = create_database(client)
    
    # Create tables
    print("\nCreating tables...")
    create_tables(client, db_name)
    
    # Create materialized views
    print("\nCreating materialized views...")
    create_materialized_views(client, db_name)
    
    # Verify setup
    if verify_setup(client, db_name):
        print("\n" + "="*60)
        print("✓ Database initialization completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Database initialization completed with errors.")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
