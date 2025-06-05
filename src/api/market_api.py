"""
Real-Time Market Intelligence Platform
Market API Module

This module provides a FastAPI application for serving market intelligence data.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import time

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import jwt
from passlib.context import CryptContext

from ..utils.logger import get_logger
from ..models.sentiment_analyzer import MarketSentimentAnalyzer

logger = get_logger(__name__)


# Models for API
class TokenData(BaseModel):
    username: str
    exp: datetime


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class PriceData(BaseModel):
    timestamp: str
    price: float
    volume: int


class SentimentData(BaseModel):
    timestamp: str
    score: float
    source: str


class NewsItem(BaseModel):
    timestamp: str
    headline: str
    source: str
    sentiment: float


class SocialPost(BaseModel):
    timestamp: str
    text: str
    user: str
    platform: str
    sentiment: float


class MarketData(BaseModel):
    symbol: str
    timerange: str
    price_data: List[PriceData]
    sentiment_data: List[SentimentData]
    news_data: List[NewsItem]
    social_data: List[SocialPost]


class MarketAPI:
    """
    FastAPI application for serving market intelligence data.
    
    This class provides endpoints for retrieving market data,
    sentiment analysis, and other market intelligence features.
    """
    
    def __init__(
        self,
        data_provider: Any = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        secret_key: str = "secret",
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30
    ):
        """
        Initialize the market API.
        
        Args:
            data_provider: Provider for market data (optional)
            host: Host to serve the API on
            port: Port to serve the API on
            secret_key: Secret key for JWT token generation
            algorithm: Algorithm for JWT token generation
            access_token_expire_minutes: Expiration time for access tokens
        """
        self.data_provider = data_provider
        self.host = host
        self.port = port
        
        # Security settings
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # OAuth2 scheme
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        # Mock users database
        self.users_db = {
            "admin": {
                "username": "admin",
                "full_name": "Administrator",
                "email": "admin@example.com",
                "hashed_password": self.get_password_hash("admin"),
                "disabled": False
            },
            "user": {
                "username": "user",
                "full_name": "Test User",
                "email": "user@example.com",
                "hashed_password": self.get_password_hash("user"),
                "disabled": False
            }
        }
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Real-Time Market Intelligence API",
            description="API for accessing real-time market intelligence data",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
        # Set up routes
        self._setup_routes()
        
        logger.info("Initialized MarketAPI")
    
    def _setup_routes(self) -> None:
        """Set up API routes."""
        
        # Authentication routes
        @self.app.post("/token", response_model=Token)
        async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
            user = self.authenticate_user(form_data.username, form_data.password)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            access_token = self.create_access_token(
                data={"sub": user.username},
                expires_delta=access_token_expires
            )
            
            return {"access_token": access_token, "token_type": "bearer"}
        
        # User routes
        @self.app.get("/users/me", response_model=User)
        async def read_users_me(current_user: User = Depends(self.get_current_user)):
            return current_user
        
        # Market data routes
        @self.app.get("/market/{symbol}", response_model=MarketData)
        async def get_market_data(
            symbol: str = Path(..., description="Stock symbol"),
            timerange: str = Query("1d", description="Time range (e.g., 1h, 1d, 1w, 1m, 3m, 1y)"),
            current_user: User = Depends(self.get_current_user)
        ):
            try:
                data = self._get_data(symbol, timerange)
                return data
            except Exception as e:
                logger.error(f"Error getting market data: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting market data: {str(e)}"
                )
        
        # Price data routes
        @self.app.get("/market/{symbol}/price", response_model=List[PriceData])
        async def get_price_data(
            symbol: str = Path(..., description="Stock symbol"),
            timerange: str = Query("1d", description="Time range (e.g., 1h, 1d, 1w, 1m, 3m, 1y)"),
            current_user: User = Depends(self.get_current_user)
        ):
            try:
                data = self._get_data(symbol, timerange)
                return data["price_data"]
            except Exception as e:
                logger.error(f"Error getting price data: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting price data: {str(e)}"
                )
        
        # Sentiment data routes
        @self.app.get("/market/{symbol}/sentiment", response_model=List[SentimentData])
        async def get_sentiment_data(
            symbol: str = Path(..., description="Stock symbol"),
            timerange: str = Query("1d", description="Time range (e.g., 1h, 1d, 1w, 1m, 3m, 1y)"),
            current_user: User = Depends(self.get_current_user)
        ):
            try:
                data = self._get_data(symbol, timerange)
                return data["sentiment_data"]
            except Exception as e:
                logger.error(f"Error getting sentiment data: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting sentiment data: {str(e)}"
                )
        
        # News data routes
        @self.app.get("/market/{symbol}/news", response_model=List[NewsItem])
        async def get_news_data(
            symbol: str = Path(..., description="Stock symbol"),
            timerange: str = Query("1d", description="Time range (e.g., 1h, 1d, 1w, 1m, 3m, 1y)"),
            current_user: User = Depends(self.get_current_user)
        ):
            try:
                data = self._get_data(symbol, timerange)
                return data["news_data"]
            except Exception as e:
                logger.error(f"Error getting news data: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting news data: {str(e)}"
                )
        
        # Social data routes
        @self.app.get("/market/{symbol}/social", response_model=List[SocialPost])
        async def get_social_data(
            symbol: str = Path(..., description="Stock symbol"),
            timerange: str = Query("1d", description="Time range (e.g., 1h, 1d, 1w, 1m, 3m, 1y)"),
            current_user: User = Depends(self.get_current_user)
        ):
            try:
                data = self._get_data(symbol, timerange)
                return data["social_data"]
            except Exception as e:
                logger.error(f"Error getting social data: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting social data: {str(e)}"
                )
        
        # Sentiment analysis routes
        @self.app.post("/analyze/sentiment")
        async def analyze_sentiment(
            text: str = Body(..., embed=True, description="Text to analyze"),
            current_user: User = Depends(self.get_current_user)
        ):
            try:
                result = self.sentiment_analyzer.analyze_market_sentiment(text)
                return result
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error analyzing sentiment: {str(e)}"
                )
        
        # Health check route
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if the password matches the hash, False otherwise
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """
        Get password hash.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """
        Get user from database.
        
        Args:
            username: Username
            
        Returns:
            User object if found, None otherwise
        """
        if username in self.users_db:
            user_dict = self.users_db[username]
            return UserInDB(**user_dict)
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create access token.
        
        Args:
            data: Data to encode in the token
            expires_delta: Expiration time
            
        Returns:
            JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        """
        Get current user from token.
        
        Args:
            token: JWT token
            
        Returns:
            User object
            
        Raises:
            HTTPException: If token is invalid or user not found
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username, exp=payload.get("exp"))
        except jwt.PyJWTError:
            raise credentials_exception
        
        user = self.get_user(username=token_data.username)
        if user is None:
            raise credentials_exception
        
        return user
    
    async def get_current_active_user(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Get current active user.
        
        Args:
            current_user: Current user
            
        Returns:
            User object
            
        Raises:
            HTTPException: If user is disabled
        """
        if current_user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    def _get_data(self, symbol: str, timerange: str) -> Dict[str, Any]:
        """
        Get data for the selected symbol and time range.
        
        Args:
            symbol: Stock symbol
            timerange: Time range (e.g., "1d", "1w", "1m")
            
        Returns:
            Dictionary with price, sentiment, news, and social data
        """
        # Use data provider if available
        if self.data_provider:
            try:
                return self.data_provider.get_data(symbol, timerange)
            except Exception as e:
                logger.error(f"Error getting data from provider: {str(e)}")
        
        # Otherwise, generate mock data
        return self._generate_mock_data(symbol, timerange)
    
    def _generate_mock_data(self, symbol: str, timerange: str) -> Dict[str, Any]:
        """
        Generate mock data for demonstration purposes.
        
        Args:
            symbol: Stock symbol
            timerange: Time range (e.g., "1d", "1w", "1m")
            
        Returns:
            Dictionary with mock price, sentiment, news, and social data
        """
        # Determine time range in hours
        hours = {
            "1h": 1,
            "1d": 24,
            "1w": 24 * 7,
            "1m": 24 * 30,
            "3m": 24 * 90,
            "1y": 24 * 365
        }.get(timerange, 24)
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Determine interval
        if hours <= 24:
            interval = timedelta(minutes=5)
        elif hours <= 24 * 7:
            interval = timedelta(hours=1)
        elif hours <= 24 * 30:
            interval = timedelta(hours=4)
        elif hours <= 24 * 90:
            interval = timedelta(days=1)
        else:
            interval = timedelta(days=7)
        
        # Generate price data
        price_data = []
        current_time = start_time
        
        # Initial price based on symbol
        base_prices = {
            "AAPL": 180.0,
            "MSFT": 350.0,
            "GOOGL": 140.0,
            "AMZN": 130.0,
            "TSLA": 250.0,
            "META": 300.0,
            "NVDA": 400.0,
            "JPM": 150.0,
            "BAC": 35.0,
            "WMT": 60.0
        }
        
        price = base_prices.get(symbol, 100.0)
        
        while current_time <= end_time:
            # Random price change
            price_change = np.random.normal(0, 0.005) * price
            price += price_change
            
            # Random volume
            volume = int(np.random.normal(1000000, 300000))
            if volume < 0:
                volume = 0
            
            # Add data point
            price_data.append({
                "timestamp": current_time.isoformat(),
                "price": round(price, 2),
                "volume": volume
            })
            
            # Move to next time
            current_time += interval
        
        # Generate sentiment data
        sentiment_data = []
        
        # Use fewer points for sentiment
        sentiment_interval = interval * 3
        current_time = start_time
        
        while current_time <= end_time:
            # Random sentiment score (-1 to 1)
            sentiment_score = np.random.normal(0, 0.3)
            if sentiment_score > 1:
                sentiment_score = 1
            elif sentiment_score < -1:
                sentiment_score = -1
            
            # Add data point
            sentiment_data.append({
                "timestamp": current_time.isoformat(),
                "score": round(sentiment_score, 2),
                "source": np.random.choice(["news", "social", "combined"])
            })
            
            # Move to next time
            current_time += sentiment_interval
        
        # Generate news data
        news_data = []
        
        # News headlines
        headlines = [
            f"{symbol} Reports Strong Quarterly Earnings",
            f"{symbol} Announces New Product Line",
            f"Analysts Upgrade {symbol} to Buy",
            f"{symbol} Expands into New Markets",
            f"CEO of {symbol} Discusses Future Growth",
            f"{symbol} Faces Regulatory Scrutiny",
            f"Market Volatility Impacts {symbol}",
            f"{symbol} Partners with Tech Giant",
            f"Investors React to {symbol} Announcement",
            f"{symbol} Stock Surges on Positive News"
        ]
        
        # Generate 5-10 news items
        num_news = np.random.randint(5, 11)
        
        for _ in range(num_news):
            # Random timestamp
            news_time = start_time + timedelta(hours=np.random.randint(0, hours))
            
            # Random headline
            headline = np.random.choice(headlines)
            
            # Random sentiment
            sentiment = np.random.normal(0, 0.5)
            if sentiment > 1:
                sentiment = 1
            elif sentiment < -1:
                sentiment = -1
            
            # Add news item
            news_data.append({
                "timestamp": news_time.isoformat(),
                "headline": headline,
                "source": np.random.choice(["Bloomberg", "Reuters", "CNBC", "WSJ"]),
                "sentiment": round(sentiment, 2)
            })
        
        # Sort news by timestamp
        news_data.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Generate social data
        social_data = []
        
        # Social posts
        post_templates = [
            f"Just bought more ${symbol}! #investing",
            f"${symbol} looking strong today. Bullish!",
            f"Not sure about ${symbol}, might sell soon.",
            f"${symbol} earnings were impressive! Long-term hold.",
            f"Disappointed with ${symbol} performance lately.",
            f"${symbol} is my top pick for this quarter.",
            f"Analysts are wrong about ${symbol}. It's undervalued.",
            f"${symbol} facing headwinds in current market.",
            f"Technical analysis shows ${symbol} ready for breakout.",
            f"${symbol} dividend increase makes it attractive."
        ]
        
        # Generate 10-20 social posts
        num_posts = np.random.randint(10, 21)
        
        for _ in range(num_posts):
            # Random timestamp
            post_time = start_time + timedelta(hours=np.random.randint(0, hours))
            
            # Random post
            post = np.random.choice(post_templates)
            
            # Random sentiment
            sentiment = np.random.normal(0, 0.7)
            if sentiment > 1:
                sentiment = 1
            elif sentiment < -1:
                sentiment = -1
            
            # Add social post
            social_data.append({
                "timestamp": post_time.isoformat(),
                "text": post,
                "user": f"user_{np.random.randint(1000, 9999)}",
                "platform": np.random.choice(["Twitter", "Reddit", "StockTwits"]),
                "sentiment": round(sentiment, 2)
            })
        
        # Sort social posts by timestamp
        social_data.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "symbol": symbol,
            "timerange": timerange,
            "price_data": price_data,
            "sentiment_data": sentiment_data,
            "news_data": news_data,
            "social_data": social_data
        }
    
    def run(self) -> None:
        """Run the API server."""
        import uvicorn
        
        logger.info(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    # Example usage
    api = MarketAPI(host="0.0.0.0", port=8000)
    api.run()

