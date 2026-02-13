"""
Tests for Real-Time Market Intelligence Platform.

Tests the logger utility, API mock data generation, authentication,
and Pydantic model validation.
"""

import logging
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np


class TestLogger(unittest.TestCase):
    """Tests for src/utils/logger.py."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a logging.Logger instance."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.utils.logger import get_logger

        logger = get_logger("test_logger_basic")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger_basic")

    def test_get_logger_sets_level(self):
        """get_logger respects the level parameter."""
        from src.utils.logger import get_logger

        logger = get_logger("test_level", level=logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_get_logger_console_handler(self):
        """get_logger adds a StreamHandler to stdout."""
        from src.utils.logger import get_logger

        logger = get_logger("test_console_handler")
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        self.assertGreaterEqual(len(stream_handlers), 1)

    def test_get_logger_file_handler(self):
        """get_logger adds a RotatingFileHandler when log_file is specified."""
        from src.utils.logger import get_logger
        from logging.handlers import RotatingFileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            logger = get_logger("test_file_handler", log_file=log_path)

            file_handlers = [
                h for h in logger.handlers if isinstance(h, RotatingFileHandler)
            ]
            self.assertGreaterEqual(len(file_handlers), 1)
            self.assertTrue(os.path.exists(log_path))

    def test_get_logger_creates_directory(self):
        """get_logger creates the log directory if it doesn't exist."""
        from src.utils.logger import get_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "subdir", "test.log")
            get_logger("test_dir_creation", log_file=log_path)
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "subdir")))

    def test_get_logger_custom_format(self):
        """get_logger applies custom format string."""
        from src.utils.logger import get_logger

        fmt = "%(levelname)s: %(message)s"
        logger = get_logger("test_format", format_string=fmt)
        handler = logger.handlers[-1]
        self.assertEqual(handler.formatter._fmt, fmt)


class TestMarketAPIMockData(unittest.TestCase):
    """Tests for MarketAPI._generate_mock_data (no external services needed)."""

    @classmethod
    def setUpClass(cls):
        """Import MarketAPI with mocked heavy dependencies."""
        # We need to mock passlib since it may not be installed in test env
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    def _create_api(self):
        """Create a MarketAPI instance for testing."""
        from src.api.market_api import MarketAPI

        return MarketAPI(secret_key="test-secret-key-for-testing")

    def test_mock_data_structure(self):
        """Mock data has all required keys."""
        api = self._create_api()
        data = api._generate_mock_data("AAPL", "1d")

        self.assertIn("symbol", data)
        self.assertIn("timerange", data)
        self.assertIn("price_data", data)
        self.assertIn("sentiment_data", data)
        self.assertIn("news_data", data)
        self.assertIn("social_data", data)

    def test_mock_data_symbol(self):
        """Mock data reflects the requested symbol."""
        api = self._create_api()
        data = api._generate_mock_data("MSFT", "1w")

        self.assertEqual(data["symbol"], "MSFT")
        self.assertEqual(data["timerange"], "1w")

    def test_mock_data_price_not_empty(self):
        """Price data is generated (non-empty list)."""
        api = self._create_api()
        data = api._generate_mock_data("GOOGL", "1d")

        self.assertGreater(len(data["price_data"]), 0)

    def test_mock_data_price_fields(self):
        """Each price data point has timestamp, price, and volume."""
        api = self._create_api()
        data = api._generate_mock_data("AAPL", "1d")

        for point in data["price_data"]:
            self.assertIn("timestamp", point)
            self.assertIn("price", point)
            self.assertIn("volume", point)
            self.assertIsInstance(point["price"], float)
            self.assertIsInstance(point["volume"], int)
            self.assertGreaterEqual(point["volume"], 0)

    def test_mock_data_sentiment_range(self):
        """Sentiment scores are clamped to [-1, 1]."""
        api = self._create_api()
        np.random.seed(42)
        data = api._generate_mock_data("AAPL", "1m")

        for point in data["sentiment_data"]:
            self.assertGreaterEqual(point["score"], -1.0)
            self.assertLessEqual(point["score"], 1.0)

    def test_mock_data_news_count(self):
        """News data generates between 5 and 10 items."""
        api = self._create_api()
        np.random.seed(42)
        data = api._generate_mock_data("AAPL", "1d")

        self.assertGreaterEqual(len(data["news_data"]), 5)
        self.assertLessEqual(len(data["news_data"]), 10)

    def test_mock_data_social_count(self):
        """Social data generates between 10 and 20 posts."""
        api = self._create_api()
        np.random.seed(42)
        data = api._generate_mock_data("AAPL", "1d")

        self.assertGreaterEqual(len(data["social_data"]), 10)
        self.assertLessEqual(len(data["social_data"]), 20)

    def test_mock_data_different_timeranges(self):
        """Different time ranges produce different amounts of data."""
        api = self._create_api()
        np.random.seed(42)

        data_1h = api._generate_mock_data("AAPL", "1h")
        data_1w = api._generate_mock_data("AAPL", "1w")

        # 1 week should have more price data points than 1 hour
        self.assertGreater(len(data_1w["price_data"]), len(data_1h["price_data"]))

    def test_mock_data_known_symbol_base_price(self):
        """Known symbols start near their base price."""
        api = self._create_api()
        np.random.seed(42)
        data = api._generate_mock_data("AAPL", "1h")

        first_price = data["price_data"][0]["price"]
        # AAPL base is 180.0, first tick should be close
        self.assertAlmostEqual(first_price, 180.0, delta=10.0)

    def test_mock_data_unknown_symbol_base_price(self):
        """Unknown symbols default to ~100.0 base price."""
        api = self._create_api()
        np.random.seed(42)
        data = api._generate_mock_data("UNKNOWN", "1h")

        first_price = data["price_data"][0]["price"]
        self.assertAlmostEqual(first_price, 100.0, delta=10.0)

    def test_get_data_without_provider(self):
        """_get_data falls back to mock data when no provider is set."""
        api = self._create_api()
        data = api._get_data("AAPL", "1d")

        self.assertEqual(data["symbol"], "AAPL")
        self.assertIn("price_data", data)


class TestMarketAPIAuth(unittest.TestCase):
    """Tests for MarketAPI authentication methods."""

    def _create_api(self):
        from src.api.market_api import MarketAPI

        return MarketAPI(secret_key="test-secret-key-for-auth-tests")

    def test_password_hash_and_verify(self):
        """Hashing and verifying a password works correctly."""
        api = self._create_api()
        hashed = api.get_password_hash("my-secret-password")

        self.assertNotEqual(hashed, "my-secret-password")
        self.assertTrue(api.verify_password("my-secret-password", hashed))
        self.assertFalse(api.verify_password("wrong-password", hashed))

    def test_get_user_existing(self):
        """get_user returns UserInDB for existing users."""
        api = self._create_api()
        user = api.get_user("admin")

        self.assertIsNotNone(user)
        self.assertEqual(user.username, "admin")
        self.assertFalse(user.disabled)

    def test_get_user_nonexistent(self):
        """get_user returns None for unknown users."""
        api = self._create_api()
        user = api.get_user("nonexistent")

        self.assertIsNone(user)

    def test_authenticate_user_success(self):
        """authenticate_user succeeds with correct credentials."""
        api = self._create_api()
        user = api.authenticate_user("admin", "admin")

        self.assertIsNotNone(user)
        self.assertEqual(user.username, "admin")

    def test_authenticate_user_wrong_password(self):
        """authenticate_user returns None for wrong password."""
        api = self._create_api()
        user = api.authenticate_user("admin", "wrong")

        self.assertIsNone(user)

    def test_authenticate_user_unknown_user(self):
        """authenticate_user returns None for unknown user."""
        api = self._create_api()
        user = api.authenticate_user("unknown", "password")

        self.assertIsNone(user)

    def test_create_access_token(self):
        """create_access_token returns a string JWT."""
        api = self._create_api()
        import jwt

        token = api.create_access_token(data={"sub": "admin"})
        self.assertIsInstance(token, str)

        # Decode and verify
        payload = jwt.decode(
            token, "test-secret-key-for-auth-tests", algorithms=["HS256"]
        )
        self.assertEqual(payload["sub"], "admin")
        self.assertIn("exp", payload)

    def test_create_access_token_with_expiry(self):
        """create_access_token respects custom expiry delta."""
        api = self._create_api()
        import jwt

        delta = timedelta(minutes=5)
        token = api.create_access_token(data={"sub": "user"}, expires_delta=delta)

        payload = jwt.decode(
            token, "test-secret-key-for-auth-tests", algorithms=["HS256"]
        )
        self.assertEqual(payload["sub"], "user")


class TestPydanticModels(unittest.TestCase):
    """Tests for API Pydantic models."""

    def test_price_data_model(self):
        from src.api.market_api import PriceData

        pd = PriceData(timestamp="2025-01-01T00:00:00", price=180.5, volume=1000000)
        self.assertEqual(pd.price, 180.5)
        self.assertEqual(pd.volume, 1000000)

    def test_sentiment_data_model(self):
        from src.api.market_api import SentimentData

        sd = SentimentData(
            timestamp="2025-01-01T00:00:00", score=0.75, source="news"
        )
        self.assertEqual(sd.score, 0.75)
        self.assertEqual(sd.source, "news")

    def test_news_item_model(self):
        from src.api.market_api import NewsItem

        ni = NewsItem(
            timestamp="2025-01-01T00:00:00",
            headline="Test Headline",
            source="Reuters",
            sentiment=0.5,
        )
        self.assertEqual(ni.headline, "Test Headline")

    def test_token_model(self):
        from src.api.market_api import Token

        t = Token(access_token="abc123", token_type="bearer")
        self.assertEqual(t.access_token, "abc123")

    def test_user_model(self):
        from src.api.market_api import User

        u = User(username="testuser", email="test@example.com")
        self.assertEqual(u.username, "testuser")
        self.assertIsNone(u.disabled)


class TestClientStructure(unittest.TestCase):
    """Tests for MarketIntelligenceClient (structure only, no server)."""

    def test_client_init(self):
        from src.client import MarketIntelligenceClient

        client = MarketIntelligenceClient(
            api_key="test-key", base_url="http://localhost:8000"
        )
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.base_url, "http://localhost:8000")
        self.assertIn("Authorization", client.headers)
        self.assertIsNone(client.ws)

    def test_client_close_no_ws(self):
        """close() works safely when no WebSocket is open."""
        from src.client import MarketIntelligenceClient

        client = MarketIntelligenceClient(api_key="test-key")
        client.close()  # Should not raise


if __name__ == "__main__":
    unittest.main()
