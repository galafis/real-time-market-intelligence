"""
Real-Time Market Intelligence Platform
Kafka Producer Module

This module provides functionality to produce market data to Kafka topics.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import random
import threading

from confluent_kafka import Producer

from ..utils.logger import get_logger

logger = get_logger(__name__)


class KafkaProducerManager:
    """
    Manager for Kafka producers that handles production of market data
    to multiple topics with automatic recovery and error handling.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: Optional[str] = None,
        acks: str = "1",  # Wait for leader to acknowledge
        compression_type: str = "gzip",
        retries: int = 5,
        retry_backoff_ms: int = 100,
        linger_ms: int = 20,  # Wait up to 20ms to batch messages
        batch_size: int = 16384  # 16KB batch size
    ):
        """
        Initialize the Kafka producer manager.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            client_id: Client ID for this producer
            acks: Acknowledgement level
            compression_type: Compression type (gzip, snappy, lz4, zstd)
            retries: Number of retries for failed messages
            retry_backoff_ms: Backoff time between retries
            linger_ms: Delay to wait for more messages before sending a batch
            batch_size: Maximum size of a batch in bytes
        """
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id or f"market_intelligence_producer_{int(time.time())}"
        
        # Producer configuration
        self.config = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": self.client_id,
            "acks": acks,
            "compression.type": compression_type,
            "retries": retries,
            "retry.backoff.ms": retry_backoff_ms,
            "linger.ms": linger_ms,
            "batch.size": batch_size
        }
        
        # Create producer
        self.producer = Producer(self.config)
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Initialized KafkaProducerManager with bootstrap servers: {bootstrap_servers}")
    
    def _delivery_report(self, err, msg) -> None:
        """
        Callback function for message delivery reports.
        
        Args:
            err: Error object (None if successful)
            msg: Message object
        """
        if err is not None:
            logger.error(f"Message delivery failed for topic {msg.topic()}: {err}")
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )
    
    def produce(
        self,
        topic: str,
        message: Union[Dict[str, Any], str],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        message_type: str = "json"
    ) -> None:
        """
        Produce a message to a Kafka topic.
        
        Args:
            topic: Topic to produce to
            message: Message payload (dict for JSON, string otherwise)
            key: Message key (optional)
            headers: Message headers (optional)
            message_type: Type of message ("json", "string")
        """
        with self.lock:
            try:
                # Serialize message based on type
                if message_type == "json":
                    if not isinstance(message, dict):
                        raise TypeError("Message must be a dict for JSON type")
                    value = json.dumps(message).encode("utf-8")
                elif message_type == "string":
                    if not isinstance(message, str):
                        raise TypeError("Message must be a string for string type")
                    value = message.encode("utf-8")
                else:
                    raise ValueError(f"Unsupported message type: {message_type}")
                
                # Produce message
                self.producer.produce(
                    topic,
                    value=value,
                    key=key.encode("utf-8") if key else None,
                    headers=headers,
                    callback=self._delivery_report
                )
                
                # Poll to trigger delivery report callback
                self.producer.poll(0)
            
            except BufferError:
                logger.warning("Producer queue is full, flushing...")
                self.producer.flush(timeout=5)  # Wait up to 5 seconds
                
                # Retry producing the message
                try:
                    self.producer.produce(
                        topic,
                        value=value,
                        key=key.encode("utf-8") if key else None,
                        headers=headers,
                        callback=self._delivery_report
                    )
                    self.producer.poll(0)
                except Exception as e:
                    logger.error(f"Failed to produce message after flush: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error producing message to topic {topic}: {str(e)}")
    
    def flush(self, timeout: float = 10.0) -> int:
        """
        Wait for all messages in the producer queue to be delivered.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            Number of messages still waiting to be delivered
        """
        with self.lock:
            logger.info(f"Flushing producer queue (timeout={timeout}s)")
            remaining = self.producer.flush(timeout=timeout)
            if remaining > 0:
                logger.warning(f"{remaining} messages still in queue after flush")
            else:
                logger.info("Producer queue flushed successfully")
            return remaining
    
    def close(self) -> None:
        """Close the producer and flush any remaining messages."""
        self.flush()
        logger.info("Closed KafkaProducerManager")


class MarketDataProducer:
    """
    Specialized producer for market data that simulates data streams
    from various financial sources.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: Optional[str] = None,
        simulation_interval: float = 1.0,  # Interval between simulated updates
        symbols: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    ):
        """
        Initialize the market data producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            client_id: Client ID for this producer
            simulation_interval: Interval for sending simulated updates in seconds
            symbols: List of symbols to simulate data for
        """
        self.producer_manager = KafkaProducerManager(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id
        )
        
        self.simulation_interval = simulation_interval
        self.symbols = symbols
        
        # Topics
        self.price_topic = "market_data_prices"
        self.news_topic = "market_data_news"
        self.social_topic = "market_data_social"
        
        # Simulation state
        self.current_prices = {symbol: random.uniform(100, 1000) for symbol in symbols}
        self.running = False
        self.thread = None
        
        logger.info("Initialized MarketDataProducer")
    
    def start_simulation(self) -> None:
        """Start the market data simulation."""
        if self.running:
            logger.warning("Simulation is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()
        
        logger.info("Started market data simulation")
    
    def stop_simulation(self) -> None:
        """Stop the market data simulation."""
        if not self.running:
            logger.warning("Simulation is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Stopped market data simulation")
    
    def _simulation_loop(self) -> None:
        """Main simulation loop."""
        logger.info("Starting simulation loop")
        
        while self.running:
            try:
                # Simulate price updates
                self._simulate_price_updates()
                
                # Simulate news updates (less frequent)
                if random.random() < 0.1:
                    self._simulate_news_update()
                
                # Simulate social media updates (more frequent)
                if random.random() < 0.3:
                    self._simulate_social_update()
                
                # Wait for the next interval
                time.sleep(self.simulation_interval)
            
            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
                time.sleep(5)  # Sleep longer on error
    
    def _simulate_price_updates(self) -> None:
        """Simulate and produce price updates for all symbols."""
        for symbol in self.symbols:
            # Simulate price change
            change_percent = random.normalvariate(0, 0.01)  # Normal distribution for change
            new_price = self.current_prices[symbol] * (1 + change_percent)
            self.current_prices[symbol] = new_price
            
            # Create message
            message = {
                "symbol": symbol,
                "price": round(new_price, 2),
                "change_percent": round(change_percent * 100, 4),
                "volume": random.randint(1000, 100000),
                "timestamp": datetime.now().isoformat()
            }
            
            # Produce message
            self.producer_manager.produce(
                topic=self.price_topic,
                message=message,
                key=symbol,
                message_type="json"
            )
    
    def _simulate_news_update(self) -> None:
        """Simulate and produce a news update."""
        symbol = random.choice(self.symbols)
        headlines = [
            f"{symbol} stock surges on positive earnings report",
            f"New product launch boosts {symbol} shares",
            f"Analysts upgrade {symbol} rating to 'Buy'",
            f"{symbol} faces regulatory scrutiny",
            f"Market volatility impacts {symbol} price"
        ]
        
        # Create message
        message = {
            "headline": random.choice(headlines),
            "source": random.choice(["Reuters", "Bloomberg", "CNBC"]),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Produce message
        self.producer_manager.produce(
            topic=self.news_topic,
            message=message,
            key=symbol,
            message_type="json"
        )
    
    def _simulate_social_update(self) -> None:
        """Simulate and produce a social media update."""
        symbol = random.choice(self.symbols)
        sentiments = ["positive", "negative", "neutral"]
        platforms = ["Twitter", "Reddit", "StockTwits"]
        
        # Create message
        message = {
            "text": f"Discussion about ${symbol} is trending {random.choice(sentiments)}",
            "platform": random.choice(platforms),
            "user": f"user_{random.randint(1000, 9999)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Produce message
        self.producer_manager.produce(
            topic=self.social_topic,
            message=message,
            key=symbol,
            message_type="json"
        )
    
    def close(self) -> None:
        """Stop simulation and close the producer."""
        self.stop_simulation()
        self.producer_manager.close()
        logger.info("Closed MarketDataProducer")


if __name__ == "__main__":
    # Example usage
    producer = MarketDataProducer(bootstrap_servers="localhost:9092")
    
    # Start simulation
    producer.start_simulation()
    
    # Keep running
    try:
        print("Simulating market data (press Ctrl+C to stop)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        producer.close()

