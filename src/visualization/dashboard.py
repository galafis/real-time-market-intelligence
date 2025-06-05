"""
Real-Time Market Intelligence Platform
Dashboard Module

This module provides functionality to create interactive dashboards
for visualizing market data and sentiment analysis.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import threading
import time

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MarketDashboard:
    """
    Interactive dashboard for visualizing market data and sentiment analysis.
    
    This class provides functionality to create and serve a Dash-based
    dashboard for real-time market intelligence visualization.
    """
    
    def __init__(
        self,
        data_provider: Any = None,
        port: int = 8050,
        debug: bool = False,
        theme: str = "darkly"
    ):
        """
        Initialize the market dashboard.
        
        Args:
            data_provider: Provider for market data (optional)
            port: Port to serve the dashboard on
            debug: Whether to run in debug mode
            theme: Bootstrap theme to use
        """
        self.data_provider = data_provider
        self.port = port
        self.debug = debug
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY if theme == "darkly" else dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Set up layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Cache for data
        self.data_cache = {
            "price_data": {},
            "sentiment_data": {},
            "news_data": {},
            "social_data": {}
        }
        
        # Thread for data updates
        self.update_thread = None
        self.running = False
        
        logger.info("Initialized MarketDashboard")
    
    def _setup_layout(self) -> None:
        """Set up the dashboard layout."""
        # Define available symbols
        self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "WMT"]
        
        # Define time ranges
        self.time_ranges = [
            {"label": "1 Hour", "value": "1h"},
            {"label": "1 Day", "value": "1d"},
            {"label": "1 Week", "value": "1w"},
            {"label": "1 Month", "value": "1m"},
            {"label": "3 Months", "value": "3m"},
            {"label": "1 Year", "value": "1y"}
        ]
        
        # Create layout
        self.app.layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    dbc.Col(
                        html.H1(
                            "Real-Time Market Intelligence Dashboard",
                            className="text-center my-4"
                        ),
                        width=12
                    )
                ),
                
                # Controls
                dbc.Row(
                    [
                        # Symbol selector
                        dbc.Col(
                            [
                                html.Label("Symbol"),
                                dcc.Dropdown(
                                    id="symbol-dropdown",
                                    options=[{"label": s, "value": s} for s in self.symbols],
                                    value="AAPL",
                                    clearable=False
                                )
                            ],
                            width=3
                        ),
                        
                        # Time range selector
                        dbc.Col(
                            [
                                html.Label("Time Range"),
                                dcc.Dropdown(
                                    id="timerange-dropdown",
                                    options=self.time_ranges,
                                    value="1d",
                                    clearable=False
                                )
                            ],
                            width=3
                        ),
                        
                        # Refresh button
                        dbc.Col(
                            [
                                html.Label("Refresh"),
                                html.Div(
                                    dbc.Button(
                                        "Refresh Data",
                                        id="refresh-button",
                                        color="primary",
                                        className="w-100"
                                    )
                                )
                            ],
                            width=3
                        ),
                        
                        # Auto-refresh toggle
                        dbc.Col(
                            [
                                html.Label("Auto-Refresh"),
                                dbc.Checklist(
                                    id="auto-refresh-toggle",
                                    options=[{"label": "Enable", "value": 1}],
                                    value=[1],
                                    switch=True
                                )
                            ],
                            width=3
                        )
                    ],
                    className="mb-4"
                ),
                
                # Price chart
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Price Chart"),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="price-chart",
                                        style={"height": "400px"}
                                    )
                                )
                            ]
                        ),
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Sentiment and volume charts
                dbc.Row(
                    [
                        # Sentiment chart
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Sentiment Analysis"),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id="sentiment-chart",
                                            style={"height": "300px"}
                                        )
                                    )
                                ]
                            ),
                            width=6
                        ),
                        
                        # Volume chart
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Trading Volume"),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id="volume-chart",
                                            style={"height": "300px"}
                                        )
                                    )
                                ]
                            ),
                            width=6
                        )
                    ],
                    className="mb-4"
                ),
                
                # News and social media
                dbc.Row(
                    [
                        # News feed
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Latest News"),
                                    dbc.CardBody(
                                        html.Div(
                                            id="news-feed",
                                            style={"height": "300px", "overflow-y": "auto"}
                                        )
                                    )
                                ]
                            ),
                            width=6
                        ),
                        
                        # Social media feed
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Social Media Sentiment"),
                                    dbc.CardBody(
                                        html.Div(
                                            id="social-feed",
                                            style={"height": "300px", "overflow-y": "auto"}
                                        )
                                    )
                                ]
                            ),
                            width=6
                        )
                    ],
                    className="mb-4"
                ),
                
                # Market summary
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Market Summary"),
                                dbc.CardBody(
                                    html.Div(id="market-summary")
                                )
                            ]
                        ),
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Hidden div for storing data
                html.Div(id="data-store", style={"display": "none"}),
                
                # Auto-refresh interval
                dcc.Interval(
                    id="auto-refresh-interval",
                    interval=10 * 1000,  # 10 seconds
                    n_intervals=0
                ),
                
                # Footer
                dbc.Row(
                    dbc.Col(
                        html.P(
                            f"© {datetime.now().year} Real-Time Market Intelligence Platform | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            className="text-center text-muted"
                        ),
                        width=12
                    )
                )
            ],
            fluid=True,
            className="p-4"
        )
    
    def _setup_callbacks(self) -> None:
        """Set up the dashboard callbacks."""
        
        # Callback for auto-refresh interval
        @self.app.callback(
            Output("auto-refresh-interval", "disabled"),
            Input("auto-refresh-toggle", "value")
        )
        def toggle_auto_refresh(value):
            return not value or len(value) == 0
        
        # Callback for refreshing data
        @self.app.callback(
            Output("data-store", "children"),
            [
                Input("refresh-button", "n_clicks"),
                Input("auto-refresh-interval", "n_intervals"),
                Input("symbol-dropdown", "value"),
                Input("timerange-dropdown", "value")
            ]
        )
        def refresh_data(n_clicks, n_intervals, symbol, timerange):
            # Get data for the selected symbol and time range
            data = self._get_data(symbol, timerange)
            
            # Convert to JSON
            return json.dumps(data)
        
        # Callback for updating price chart
        @self.app.callback(
            Output("price-chart", "figure"),
            Input("data-store", "children")
        )
        def update_price_chart(data_json):
            # Parse data
            data = json.loads(data_json) if data_json else {}
            
            # Get price data
            price_data = data.get("price_data", [])
            
            # Create figure
            fig = self._create_price_chart(price_data)
            
            return fig
        
        # Callback for updating sentiment chart
        @self.app.callback(
            Output("sentiment-chart", "figure"),
            Input("data-store", "children")
        )
        def update_sentiment_chart(data_json):
            # Parse data
            data = json.loads(data_json) if data_json else {}
            
            # Get sentiment data
            sentiment_data = data.get("sentiment_data", [])
            
            # Create figure
            fig = self._create_sentiment_chart(sentiment_data)
            
            return fig
        
        # Callback for updating volume chart
        @self.app.callback(
            Output("volume-chart", "figure"),
            Input("data-store", "children")
        )
        def update_volume_chart(data_json):
            # Parse data
            data = json.loads(data_json) if data_json else {}
            
            # Get price data (contains volume)
            price_data = data.get("price_data", [])
            
            # Create figure
            fig = self._create_volume_chart(price_data)
            
            return fig
        
        # Callback for updating news feed
        @self.app.callback(
            Output("news-feed", "children"),
            Input("data-store", "children")
        )
        def update_news_feed(data_json):
            # Parse data
            data = json.loads(data_json) if data_json else {}
            
            # Get news data
            news_data = data.get("news_data", [])
            
            # Create news feed
            return self._create_news_feed(news_data)
        
        # Callback for updating social feed
        @self.app.callback(
            Output("social-feed", "children"),
            Input("data-store", "children")
        )
        def update_social_feed(data_json):
            # Parse data
            data = json.loads(data_json) if data_json else {}
            
            # Get social data
            social_data = data.get("social_data", [])
            
            # Create social feed
            return self._create_social_feed(social_data)
        
        # Callback for updating market summary
        @self.app.callback(
            Output("market-summary", "children"),
            Input("data-store", "children")
        )
        def update_market_summary(data_json):
            # Parse data
            data = json.loads(data_json) if data_json else {}
            
            # Get all data
            price_data = data.get("price_data", [])
            sentiment_data = data.get("sentiment_data", [])
            
            # Create market summary
            return self._create_market_summary(price_data, sentiment_data)
    
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
    
    def _create_price_chart(self, price_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a price chart figure.
        
        Args:
            price_data: List of price data points
            
        Returns:
            Plotly figure object
        """
        if not price_data:
            # Return empty figure
            return go.Figure()
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["price"],
                mode="lines",
                name="Price",
                line=dict(color="#2FA4E7", width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            title=None,
            xaxis_title=None,
            yaxis_title="Price ($)",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)"
        )
        
        return fig
    
    def _create_sentiment_chart(self, sentiment_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a sentiment chart figure.
        
        Args:
            sentiment_data: List of sentiment data points
            
        Returns:
            Plotly figure object
        """
        if not sentiment_data:
            # Return empty figure
            return go.Figure()
        
        # Convert to DataFrame
        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create figure
        fig = go.Figure()
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["score"],
                mode="lines+markers",
                name="Sentiment",
                line=dict(color="#2FA4E7", width=2),
                marker=dict(
                    color=df["score"].apply(
                        lambda x: "green" if x > 0.1 else "red" if x < -0.1 else "gray"
                    ),
                    size=8
                )
            )
        )
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Neutral",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            title=None,
            xaxis_title=None,
            yaxis_title="Sentiment Score",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)",
                range=[-1.1, 1.1]
            ),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)"
        )
        
        return fig
    
    def _create_volume_chart(self, price_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a volume chart figure.
        
        Args:
            price_data: List of price data points (contains volume)
            
        Returns:
            Plotly figure object
        """
        if not price_data:
            # Return empty figure
            return go.Figure()
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create figure
        fig = go.Figure()
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                marker_color="#73C6B6"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=None,
            xaxis_title=None,
            yaxis_title="Volume",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)"
        )
        
        return fig
    
    def _create_news_feed(self, news_data: List[Dict[str, Any]]) -> List:
        """
        Create a news feed component.
        
        Args:
            news_data: List of news items
            
        Returns:
            List of Dash components
        """
        if not news_data:
            return [html.P("No news data available")]
        
        # Create news items
        news_items = []
        
        for item in news_data:
            # Determine sentiment color
            sentiment = item.get("sentiment", 0)
            if sentiment > 0.1:
                sentiment_color = "success"
                sentiment_icon = "📈"
            elif sentiment < -0.1:
                sentiment_color = "danger"
                sentiment_icon = "📉"
            else:
                sentiment_color = "secondary"
                sentiment_icon = "📊"
            
            # Format timestamp
            timestamp = datetime.fromisoformat(item["timestamp"])
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
            
            # Create news item
            news_item = dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(item["headline"], className="card-title"),
                        html.P(
                            [
                                f"Source: {item.get('source', 'Unknown')} | ",
                                f"Time: {timestamp_str} | ",
                                html.Span(
                                    f"{sentiment_icon} Sentiment: {sentiment:.2f}",
                                    className=f"text-{sentiment_color}"
                                )
                            ],
                            className="card-text text-muted"
                        )
                    ]
                ),
                className="mb-3"
            )
            
            news_items.append(news_item)
        
        return news_items
    
    def _create_social_feed(self, social_data: List[Dict[str, Any]]) -> List:
        """
        Create a social media feed component.
        
        Args:
            social_data: List of social media posts
            
        Returns:
            List of Dash components
        """
        if not social_data:
            return [html.P("No social media data available")]
        
        # Create social items
        social_items = []
        
        for item in social_data:
            # Determine sentiment color
            sentiment = item.get("sentiment", 0)
            if sentiment > 0.1:
                sentiment_color = "success"
                sentiment_icon = "📈"
            elif sentiment < -0.1:
                sentiment_color = "danger"
                sentiment_icon = "📉"
            else:
                sentiment_color = "secondary"
                sentiment_icon = "📊"
            
            # Format timestamp
            timestamp = datetime.fromisoformat(item["timestamp"])
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
            
            # Create social item
            social_item = dbc.Card(
                dbc.CardBody(
                    [
                        html.P(item["text"], className="card-text"),
                        html.P(
                            [
                                f"User: {item.get('user', 'Anonymous')} | ",
                                f"Platform: {item.get('platform', 'Unknown')} | ",
                                f"Time: {timestamp_str} | ",
                                html.Span(
                                    f"{sentiment_icon} Sentiment: {sentiment:.2f}",
                                    className=f"text-{sentiment_color}"
                                )
                            ],
                            className="card-text text-muted"
                        )
                    ]
                ),
                className="mb-3"
            )
            
            social_items.append(social_item)
        
        return social_items
    
    def _create_market_summary(
        self,
        price_data: List[Dict[str, Any]],
        sentiment_data: List[Dict[str, Any]]
    ) -> List:
        """
        Create a market summary component.
        
        Args:
            price_data: List of price data points
            sentiment_data: List of sentiment data points
            
        Returns:
            List of Dash components
        """
        if not price_data:
            return [html.P("No market data available")]
        
        # Calculate price metrics
        df_price = pd.DataFrame(price_data)
        df_price["timestamp"] = pd.to_datetime(df_price["timestamp"])
        
        # Sort by timestamp
        df_price = df_price.sort_values("timestamp")
        
        # Get current price
        current_price = df_price["price"].iloc[-1]
        
        # Calculate price change
        if len(df_price) > 1:
            previous_price = df_price["price"].iloc[-2]
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
        else:
            price_change = 0
            price_change_pct = 0
        
        # Calculate average volume
        avg_volume = df_price["volume"].mean()
        
        # Calculate sentiment metrics
        if sentiment_data:
            df_sentiment = pd.DataFrame(sentiment_data)
            avg_sentiment = df_sentiment["score"].mean()
            
            # Determine sentiment trend
            if avg_sentiment > 0.1:
                sentiment_trend = "Positive"
                sentiment_color = "success"
            elif avg_sentiment < -0.1:
                sentiment_trend = "Negative"
                sentiment_color = "danger"
            else:
                sentiment_trend = "Neutral"
                sentiment_color = "secondary"
        else:
            avg_sentiment = 0
            sentiment_trend = "Neutral"
            sentiment_color = "secondary"
        
        # Create summary
        summary = [
            dbc.Row(
                [
                    # Price metrics
                    dbc.Col(
                        [
                            html.H4("Price Metrics"),
                            html.P(f"Current Price: ${current_price:.2f}"),
                            html.P(
                                [
                                    "Price Change: ",
                                    html.Span(
                                        f"${price_change:.2f} ({price_change_pct:.2f}%)",
                                        className=f"text-{'success' if price_change >= 0 else 'danger'}"
                                    )
                                ]
                            ),
                            html.P(f"Average Volume: {int(avg_volume):,}")
                        ],
                        width=4
                    ),
                    
                    # Sentiment metrics
                    dbc.Col(
                        [
                            html.H4("Sentiment Metrics"),
                            html.P(f"Average Sentiment: {avg_sentiment:.2f}"),
                            html.P(
                                [
                                    "Sentiment Trend: ",
                                    html.Span(
                                        sentiment_trend,
                                        className=f"text-{sentiment_color}"
                                    )
                                ]
                            )
                        ],
                        width=4
                    ),
                    
                    # Market recommendation
                    dbc.Col(
                        [
                            html.H4("Market Recommendation"),
                            html.P(
                                self._generate_market_recommendation(
                                    price_change_pct, avg_sentiment
                                ),
                                className="lead"
                            )
                        ],
                        width=4
                    )
                ]
            )
        ]
        
        return summary
    
    def _generate_market_recommendation(
        self,
        price_change_pct: float,
        avg_sentiment: float
    ) -> str:
        """
        Generate a market recommendation based on price change and sentiment.
        
        Args:
            price_change_pct: Price change percentage
            avg_sentiment: Average sentiment score
            
        Returns:
            Market recommendation string
        """
        # Combine price change and sentiment
        if price_change_pct > 2 and avg_sentiment > 0.3:
            return "Strong Buy - Positive momentum and sentiment"
        elif price_change_pct > 1 and avg_sentiment > 0.1:
            return "Buy - Positive indicators"
        elif price_change_pct < -2 and avg_sentiment < -0.3:
            return "Strong Sell - Negative momentum and sentiment"
        elif price_change_pct < -1 and avg_sentiment < -0.1:
            return "Sell - Negative indicators"
        elif abs(price_change_pct) < 0.5 and abs(avg_sentiment) < 0.1:
            return "Hold - Stable market conditions"
        elif price_change_pct > 0 and avg_sentiment < -0.2:
            return "Caution - Price up but sentiment negative"
        elif price_change_pct < 0 and avg_sentiment > 0.2:
            return "Watch - Price down but sentiment positive"
        else:
            return "Neutral - Mixed signals"
    
    def start_data_updates(self, interval: float = 10.0) -> None:
        """
        Start a background thread for data updates.
        
        Args:
            interval: Update interval in seconds
        """
        if self.running:
            logger.warning("Data updates already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(
            target=self._data_update_loop,
            args=(interval,),
            daemon=True
        )
        self.update_thread.start()
        
        logger.info(f"Started data updates with interval {interval}s")
    
    def stop_data_updates(self) -> None:
        """Stop the background data update thread."""
        if not self.running:
            logger.warning("Data updates not running")
            return
        
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("Stopped data updates")
    
    def _data_update_loop(self, interval: float) -> None:
        """
        Background loop for updating data.
        
        Args:
            interval: Update interval in seconds
        """
        logger.info("Starting data update loop")
        
        while self.running:
            try:
                # Update data for all symbols
                for symbol in self.symbols:
                    # Update data for different time ranges
                    for time_range in ["1h", "1d", "1w"]:
                        data = self._get_data(symbol, time_range)
                        
                        # Update cache
                        cache_key = f"{symbol}_{time_range}"
                        self.data_cache[cache_key] = data
                
                logger.debug("Updated data cache")
            
            except Exception as e:
                logger.error(f"Error in data update loop: {str(e)}")
            
            # Sleep for the specified interval
            time.sleep(interval)
    
    def run(self) -> None:
        """Run the dashboard server."""
        # Start data updates
        self.start_data_updates()
        
        try:
            # Run the server
            self.app.run_server(
                debug=self.debug,
                port=self.port,
                host="0.0.0.0"
            )
        finally:
            # Stop data updates
            self.stop_data_updates()


if __name__ == "__main__":
    # Example usage
    dashboard = MarketDashboard(port=8050, debug=True)
    dashboard.run()

