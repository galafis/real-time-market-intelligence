"""Market Intelligence Client

Client library for interacting with the Real-Time Market Intelligence Platform.
Provides a simple interface for subscribing to asset updates, sentiment analysis,
and price forecasting.

Author: galafis
"""

import asyncio
import json
from typing import List, Dict, Callable, Optional
import websocket
import requests


class MarketIntelligenceClient:
    """Client for accessing the Market Intelligence Platform API."""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        """
        Initialize the Market Intelligence Client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of the API server
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.ws = None
    
    def subscribe_to_assets(self, symbols: List[str], callback: Callable[[Dict], None]):
        """
        Subscribe to real-time updates for specified assets.
        
        Args:
            symbols: List of asset symbols to monitor (e.g., ['AAPL', 'MSFT'])
            callback: Function to call when updates are received
        """
        ws_url = f"{self.base_url.replace('http', 'ws')}/ws/assets"
        
        def on_message(ws, message):
            data = json.loads(message)
            callback(data)
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        def on_open(ws):
            # Subscribe to symbols
            subscribe_msg = {
                "action": "subscribe",
                "symbols": symbols
            }
            ws.send(json.dumps(subscribe_msg))
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            header={"Authorization": f"Bearer {self.api_key}"},
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket in a separate thread
        import threading
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get market sentiment analysis for an asset.
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary containing sentiment score and sources
        """
        url = f"{self.base_url}/api/v1/sentiment/{symbol}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_price_forecast(self, symbol: str, days: int = 7) -> Dict:
        """
        Get price forecast for an asset.
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL')
            days: Number of days to forecast
            
        Returns:
            Dictionary containing date-wise price predictions
        """
        url = f"{self.base_url}/api/v1/forecast/{symbol}"
        params = {"days": days}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_asset_data(self, symbol: str, timeframe: str = "1d") -> Dict:
        """
        Get historical and real-time data for an asset.
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL')
            timeframe: Data timeframe ('1m', '5m', '1h', '1d')
            
        Returns:
            Dictionary containing asset data
        """
        url = f"{self.base_url}/api/v1/assets/{symbol}"
        params = {"timeframe": timeframe}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close WebSocket connection if open."""
        if self.ws:
            self.ws.close()
