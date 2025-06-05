"""
Real-Time Market Intelligence Platform
Kafka Consumer Module

This module provides functionality to consume market data from Kafka topics
and process it in real-time.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import json
import logging
import threading
import time
from typing import Dict, List, Callable, Optional, Any, Union
from datetime import datetime

from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from ..utils.logger import get_logger

logger = get_logger(__name__)


class KafkaConsumerManager:
    """
    Manager for Kafka consumers that handles consumption of market data
    from multiple topics with automatic recovery and error handling.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "market_intelligence_group",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        max_poll_interval_ms: int = 300000,
        session_timeout_ms: int = 30000,
        client_id: Optional[str] = None,
        max_retries: int = 5,
        retry_backoff_ms: int = 1000
    ):
        """
        Initialize the Kafka consumer manager.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            group_id: Consumer group ID
            auto_offset_reset: Where to start consuming from if no offset is stored
            enable_auto_commit: Whether to automatically commit offsets
            max_poll_interval_ms: Maximum time between polls before consumer is considered dead
            session_timeout_ms: Timeout for consumer session
            client_id: Client ID for this consumer
            max_retries: Maximum number of retries for failed operations
            retry_backoff_ms: Backoff time between retries in milliseconds
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.max_poll_interval_ms = max_poll_interval_ms
        self.session_timeout_ms = session_timeout_ms
        self.client_id = client_id or f"market_intelligence_consumer_{int(time.time())}"
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        
        # Consumer configuration
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': auto_offset_reset,
            'enable.auto.commit': enable_auto_commit,
            'max.poll.interval.ms': max_poll_interval_ms,
            'session.timeout.ms': session_timeout_ms,
            'client.id': self.client_id
        }
        
        # Admin client for topic management
        self.admin_client = AdminClient({'bootstrap.servers': bootstrap_servers})
        
        # Active consumers
        self.consumers = {}
        self.consumer_threads = {}
        self.running = {}
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Initialized KafkaConsumerManager with bootstrap servers: {bootstrap_servers}")
    
    def create_topics(self, topics: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Create Kafka topics if they don't exist.
        
        Args:
            topics: List of topic configurations with keys:
                   - name: Topic name
                   - num_partitions: Number of partitions
                   - replication_factor: Replication factor
        
        Returns:
            Dictionary mapping topic names to creation success status
        """
        new_topics = [
            NewTopic(
                topic['name'],
                num_partitions=topic.get('num_partitions', 1),
                replication_factor=topic.get('replication_factor', 1),
                config=topic.get('config', {})
            )
            for topic in topics
        ]
        
        try:
            # Create topics
            futures = self.admin_client.create_topics(new_topics)
            
            # Wait for completion
            results = {}
            for topic, future in futures.items():
                try:
                    future.result()  # Wait for result
                    results[topic] = True
                    logger.info(f"Topic {topic} created successfully")
                except Exception as e:
                    if "already exists" in str(e):
                        results[topic] = True
                        logger.info(f"Topic {topic} already exists")
                    else:
                        results[topic] = False
                        logger.error(f"Failed to create topic {topic}: {str(e)}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error creating topics: {str(e)}")
            return {topic['name']: False for topic in topics}
    
    def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None],
        message_type: str = "json",
        poll_timeout: float = 1.0,
        batch_size: int = 100,
        batch_timeout_ms: int = 1000
    ) -> bool:
        """
        Subscribe to a Kafka topic and process messages with the provided callback.
        
        Args:
            topic: Topic to subscribe to
            callback: Callback function to process messages
            message_type: Type of message ("json", "avro", "string")
            poll_timeout: Timeout for polling in seconds
            batch_size: Maximum number of messages to process in a batch
            batch_timeout_ms: Maximum time to wait for a batch in milliseconds
        
        Returns:
            True if subscription was successful, False otherwise
        """
        with self.lock:
            if topic in self.consumers:
                logger.warning(f"Already subscribed to topic {topic}")
                return False
            
            try:
                # Create consumer
                consumer = Consumer(self.config)
                
                # Subscribe to topic
                consumer.subscribe([topic])
                
                # Store consumer
                self.consumers[topic] = consumer
                self.running[topic] = True
                
                # Start consumer thread
                thread = threading.Thread(
                    target=self._consume_loop,
                    args=(topic, consumer, callback, message_type, poll_timeout, batch_size, batch_timeout_ms),
                    daemon=True
                )
                thread.start()
                
                self.consumer_threads[topic] = thread
                
                logger.info(f"Subscribed to topic {topic}")
                return True
            
            except Exception as e:
                logger.error(f"Error subscribing to topic {topic}: {str(e)}")
                return False
    
    def _consume_loop(
        self,
        topic: str,
        consumer: Consumer,
        callback: Callable[[Dict[str, Any]], None],
        message_type: str,
        poll_timeout: float,
        batch_size: int,
        batch_timeout_ms: int
    ) -> None:
        """
        Main consumption loop for a topic.
        
        Args:
            topic: Topic being consumed
            consumer: Kafka consumer
            callback: Callback function to process messages
            message_type: Type of message
            poll_timeout: Timeout for polling in seconds
            batch_size: Maximum number of messages to process in a batch
            batch_timeout_ms: Maximum time to wait for a batch in milliseconds
        """
        logger.info(f"Starting consumption loop for topic {topic}")
        
        # Batch processing variables
        messages = []
        last_batch_time = time.time() * 1000  # Convert to milliseconds
        
        try:
            while self.running.get(topic, False):
                # Poll for message
                msg = consumer.poll(timeout=poll_timeout)
                
                if msg is None:
                    # No message, check if we should process the current batch
                    current_time = time.time() * 1000
                    if messages and (current_time - last_batch_time) >= batch_timeout_ms:
                        self._process_batch(messages, callback, message_type)
                        messages = []
                        last_batch_time = current_time
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition, not an error
                        logger.debug(f"Reached end of partition for topic {topic}")
                    else:
                        # Actual error
                        logger.error(f"Error consuming from topic {topic}: {msg.error()}")
                    continue
                
                # Add message to batch
                messages.append(msg)
                
                # Process batch if full
                if len(messages) >= batch_size:
                    self._process_batch(messages, callback, message_type)
                    messages = []
                    last_batch_time = time.time() * 1000
        
        except Exception as e:
            logger.error(f"Error in consumption loop for topic {topic}: {str(e)}")
        
        finally:
            # Process any remaining messages
            if messages:
                self._process_batch(messages, callback, message_type)
            
            # Close consumer
            try:
                consumer.close()
                logger.info(f"Closed consumer for topic {topic}")
            except Exception as e:
                logger.error(f"Error closing consumer for topic {topic}: {str(e)}")
    
    def _process_batch(
        self,
        messages: List[Any],
        callback: Callable[[Dict[str, Any]], None],
        message_type: str
    ) -> None:
        """
        Process a batch of messages.
        
        Args:
            messages: List of messages to process
            callback: Callback function to process messages
            message_type: Type of message
        """
        try:
            for msg in messages:
                # Parse message based on type
                if message_type == "json":
                    try:
                        value = json.loads(msg.value().decode('utf-8'))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON message: {msg.value()}")
                        continue
                elif message_type == "string":
                    value = msg.value().decode('utf-8')
                elif message_type == "avro":
                    # Avro deserialization would go here
                    logger.warning("Avro deserialization not implemented")
                    continue
                else:
                    value = msg.value()
                
                # Add metadata
                metadata = {
                    'topic': msg.topic(),
                    'partition': msg.partition(),
                    'offset': msg.offset(),
                    'timestamp': msg.timestamp(),
                    'headers': msg.headers(),
                    'key': msg.key().decode('utf-8') if msg.key() else None,
                    'processed_at': datetime.now().isoformat()
                }
                
                # Add metadata to value if it's a dict
                if isinstance(value, dict):
                    value['__metadata'] = metadata
                
                # Process message
                try:
                    callback(value)
                except Exception as e:
                    logger.error(f"Error in callback function: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    
    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a Kafka topic.
        
        Args:
            topic: Topic to unsubscribe from
        
        Returns:
            True if unsubscription was successful, False otherwise
        """
        with self.lock:
            if topic not in self.consumers:
                logger.warning(f"Not subscribed to topic {topic}")
                return False
            
            try:
                # Stop consumer thread
                self.running[topic] = False
                
                # Wait for thread to finish
                if topic in self.consumer_threads:
                    self.consumer_threads[topic].join(timeout=5)
                
                # Close consumer
                if topic in self.consumers:
                    self.consumers[topic].close()
                
                # Remove from dictionaries
                del self.consumers[topic]
                if topic in self.consumer_threads:
                    del self.consumer_threads[topic]
                if topic in self.running:
                    del self.running[topic]
                
                logger.info(f"Unsubscribed from topic {topic}")
                return True
            
            except Exception as e:
                logger.error(f"Error unsubscribing from topic {topic}: {str(e)}")
                return False
    
    def close(self) -> None:
        """Close all consumers and stop all threads."""
        with self.lock:
            # Stop all consumer threads
            for topic in list(self.running.keys()):
                self.running[topic] = False
            
            # Close all consumers
            for topic, consumer in list(self.consumers.items()):
                try:
                    consumer.close()
                except Exception as e:
                    logger.error(f"Error closing consumer for topic {topic}: {str(e)}")
            
            # Wait for all threads to finish
            for topic, thread in list(self.consumer_threads.items()):
                thread.join(timeout=5)
            
            # Clear dictionaries
            self.consumers.clear()
            self.consumer_threads.clear()
            self.running.clear()
            
            logger.info("Closed all consumers")


class MarketDataConsumer:
    """
    Specialized consumer for market data that handles different types of
    financial data streams and provides higher-level abstractions.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "market_data_group",
        client_id: Optional[str] = None
    ):
        """
        Initialize the market data consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            group_id: Consumer group ID
            client_id: Client ID for this consumer
        """
        self.consumer_manager = KafkaConsumerManager(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            client_id=client_id,
            auto_offset_reset="latest",  # Always get latest market data
            enable_auto_commit=True,
            max_poll_interval_ms=300000,
            session_timeout_ms=30000
        )
        
        # Callbacks for different data types
        self.price_callbacks = []
        self.news_callbacks = []
        self.social_callbacks = []
        self.sentiment_callbacks = []
        
        # Topics
        self.price_topic = "market_data_prices"
        self.news_topic = "market_data_news"
        self.social_topic = "market_data_social"
        self.sentiment_topic = "market_data_sentiment"
        
        # Create topics if they don't exist
        self._create_topics()
        
        logger.info("Initialized MarketDataConsumer")
    
    def _create_topics(self) -> None:
        """Create required topics if they don't exist."""
        topics = [
            {
                'name': self.price_topic,
                'num_partitions': 8,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(24 * 60 * 60 * 1000),  # 24 hours
                    'cleanup.policy': 'delete'
                }
            },
            {
                'name': self.news_topic,
                'num_partitions': 4,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'cleanup.policy': 'delete'
                }
            },
            {
                'name': self.social_topic,
                'num_partitions': 4,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(3 * 24 * 60 * 60 * 1000),  # 3 days
                    'cleanup.policy': 'delete'
                }
            },
            {
                'name': self.sentiment_topic,
                'num_partitions': 4,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'cleanup.policy': 'delete'
                }
            }
        ]
        
        self.consumer_manager.create_topics(topics)
    
    def subscribe_to_price_updates(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to real-time price updates.
        
        Args:
            callback: Function to call with price updates
        """
        self.price_callbacks.append(callback)
        
        # Subscribe if this is the first callback
        if len(self.price_callbacks) == 1:
            self.consumer_manager.subscribe(
                topic=self.price_topic,
                callback=self._handle_price_update,
                message_type="json",
                batch_size=100,
                batch_timeout_ms=100  # Low latency for price updates
            )
            logger.info("Subscribed to price updates")
    
    def subscribe_to_news(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to financial news updates.
        
        Args:
            callback: Function to call with news updates
        """
        self.news_callbacks.append(callback)
        
        # Subscribe if this is the first callback
        if len(self.news_callbacks) == 1:
            self.consumer_manager.subscribe(
                topic=self.news_topic,
                callback=self._handle_news_update,
                message_type="json",
                batch_size=20,
                batch_timeout_ms=500  # News can have higher latency
            )
            logger.info("Subscribed to news updates")
    
    def subscribe_to_social_media(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to social media updates related to financial markets.
        
        Args:
            callback: Function to call with social media updates
        """
        self.social_callbacks.append(callback)
        
        # Subscribe if this is the first callback
        if len(self.social_callbacks) == 1:
            self.consumer_manager.subscribe(
                topic=self.social_topic,
                callback=self._handle_social_update,
                message_type="json",
                batch_size=50,
                batch_timeout_ms=300  # Social media can have medium latency
            )
            logger.info("Subscribed to social media updates")
    
    def subscribe_to_sentiment_analysis(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to sentiment analysis updates.
        
        Args:
            callback: Function to call with sentiment analysis updates
        """
        self.sentiment_callbacks.append(callback)
        
        # Subscribe if this is the first callback
        if len(self.sentiment_callbacks) == 1:
            self.consumer_manager.subscribe(
                topic=self.sentiment_topic,
                callback=self._handle_sentiment_update,
                message_type="json",
                batch_size=20,
                batch_timeout_ms=500  # Sentiment analysis can have higher latency
            )
            logger.info("Subscribed to sentiment analysis updates")
    
    def _handle_price_update(self, data: Dict[str, Any]) -> None:
        """
        Handle price update from Kafka.
        
        Args:
            data: Price update data
        """
        for callback in self.price_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in price callback: {str(e)}")
    
    def _handle_news_update(self, data: Dict[str, Any]) -> None:
        """
        Handle news update from Kafka.
        
        Args:
            data: News update data
        """
        for callback in self.news_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in news callback: {str(e)}")
    
    def _handle_social_update(self, data: Dict[str, Any]) -> None:
        """
        Handle social media update from Kafka.
        
        Args:
            data: Social media update data
        """
        for callback in self.social_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in social callback: {str(e)}")
    
    def _handle_sentiment_update(self, data: Dict[str, Any]) -> None:
        """
        Handle sentiment analysis update from Kafka.
        
        Args:
            data: Sentiment analysis update data
        """
        for callback in self.sentiment_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in sentiment callback: {str(e)}")
    
    def close(self) -> None:
        """Close all consumers and stop all threads."""
        self.consumer_manager.close()
        logger.info("Closed MarketDataConsumer")


if __name__ == "__main__":
    # Example usage
    def print_price_update(data):
        print(f"Price update: {data['symbol']} = {data['price']}")
    
    def print_news_update(data):
        print(f"News: {data['headline']}")
    
    # Create consumer
    consumer = MarketDataConsumer(bootstrap_servers="localhost:9092")
    
    # Subscribe to updates
    consumer.subscribe_to_price_updates(print_price_update)
    consumer.subscribe_to_news(print_news_update)
    
    # Keep running
    try:
        print("Listening for updates (press Ctrl+C to stop)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        consumer.close()

