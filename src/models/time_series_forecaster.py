"""
Real-Time Market Intelligence Platform
Time Series Forecaster Module

This module provides time series forecasting functionality for market data.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
from prophet import Prophet

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesForecaster:
    """
    Time series forecasting for market data.
    
    This class provides methods for forecasting market data using
    various time series forecasting models, including LSTM and Prophet.
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        lookback: int = 60,
        forecast_horizon: int = 30,
        model_type: str = "lstm"
    ):
        """
        Initialize the time series forecaster.
        
        Args:
            model_dir: Directory to store models
            lookback: Number of past time steps to use for prediction
            forecast_horizon: Number of future time steps to predict
            model_type: Type of model to use (lstm, prophet)
        """
        self.model_dir = model_dir
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        
        logger.info(f"Initialized TimeSeriesForecaster with model_type={model_type}")
    
    def _prepare_lstm_data(
        self,
        data: pd.DataFrame,
        target_col: str,
        lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            data: DataFrame with time series data
            target_col: Column to predict
            lookback: Number of past time steps to use for prediction
            
        Returns:
            X and y arrays for LSTM model
        """
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[[target_col]])
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def _build_lstm_model(
        self,
        lookback: int,
        units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ) -> Model:
        """
        Build LSTM model.
        
        Args:
            lookback: Number of past time steps to use for prediction
            units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            LSTM model
        """
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=units, return_sequences=True, input_shape=(lookback, 1)))
        model.add(Dropout(dropout))
        
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mean_squared_error"
        )
        
        return model
    
    def train_lstm(
        self,
        data: pd.DataFrame,
        symbol: str,
        target_col: str = "price",
        test_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        patience: int = 10,
        units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Train LSTM model.
        
        Args:
            data: DataFrame with time series data
            symbol: Symbol to train model for
            target_col: Column to predict
            test_size: Fraction of data to use for testing
            epochs: Number of epochs to train for
            batch_size: Batch size
            validation_split: Fraction of training data to use for validation
            patience: Number of epochs to wait for improvement before early stopping
            units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training LSTM model for {symbol}")
        
        # Prepare data
        X, y, scaler = self._prepare_lstm_data(data, target_col, self.lookback)
        
        # Split into train and test sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model
        model = self._build_lstm_model(
            lookback=self.lookback,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate
        )
        
        # Callbacks
        model_path = os.path.join(self.model_dir, f"lstm_{symbol}.h5")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        
        # Save model and scaler
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{symbol}.pkl")
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"LSTM model for {symbol} trained successfully. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return {
            "symbol": symbol,
            "model_type": "lstm",
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "history": history.history
        }
    
    def train_prophet(
        self,
        data: pd.DataFrame,
        symbol: str,
        target_col: str = "price",
        date_col: str = "timestamp",
        test_size: float = 0.2,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05
    ) -> Dict[str, Any]:
        """
        Train Prophet model.
        
        Args:
            data: DataFrame with time series data
            symbol: Symbol to train model for
            target_col: Column to predict
            date_col: Column with dates
            test_size: Fraction of data to use for testing
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            changepoint_prior_scale: Flexibility of trend
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training Prophet model for {symbol}")
        
        # Prepare data for Prophet
        prophet_data = data[[date_col, target_col]].copy()
        prophet_data.columns = ["ds", "y"]
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(prophet_data["ds"]):
            prophet_data["ds"] = pd.to_datetime(prophet_data["ds"])
        
        # Split into train and test sets
        train_size = int(len(prophet_data) * (1 - test_size))
        train_data = prophet_data[:train_size]
        test_data = prophet_data[train_size:]
        
        # Build and train model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale
        )
        
        model.fit(train_data)
        
        # Make predictions for test set
        future = model.make_future_dataframe(periods=len(test_data))
        forecast = model.predict(future)
        
        # Extract predictions for test period
        test_forecast = forecast.iloc[-len(test_data):]
        
        # Calculate metrics
        y_true = test_data["y"].values
        y_pred = test_forecast["yhat"].values
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Save model
        model_path = os.path.join(self.model_dir, f"prophet_{symbol}.pkl")
        with open(model_path, "wb") as f:
            joblib.dump(model, f)
        
        # Store model
        self.models[symbol] = model
        
        logger.info(f"Prophet model for {symbol} trained successfully. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return {
            "symbol": symbol,
            "model_type": "prophet",
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "model_path": model_path,
            "forecast": forecast,
            "components": model.plot_components(forecast)
        }
    
    def train(
        self,
        data: pd.DataFrame,
        symbol: str,
        target_col: str = "price",
        date_col: str = "timestamp",
        model_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            data: DataFrame with time series data
            symbol: Symbol to train model for
            target_col: Column to predict
            date_col: Column with dates
            model_type: Type of model to use (lstm, prophet)
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Dictionary with training results
        """
        # Use specified model type or default
        model_type = model_type or self.model_type
        
        if model_type == "lstm":
            return self.train_lstm(data, symbol, target_col=target_col, **kwargs)
        elif model_type == "prophet":
            return self.train_prophet(data, symbol, target_col=target_col, date_col=date_col, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def load_model(
        self,
        symbol: str,
        model_type: Optional[str] = None
    ) -> Any:
        """
        Load model for symbol.
        
        Args:
            symbol: Symbol to load model for
            model_type: Type of model to load (lstm, prophet)
            
        Returns:
            Loaded model
        """
        # Use specified model type or default
        model_type = model_type or self.model_type
        
        if model_type == "lstm":
            # Load model
            model_path = os.path.join(self.model_dir, f"lstm_{symbol}.h5")
            model = load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, f"scaler_{symbol}.pkl")
            scaler = joblib.load(scaler_path)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            return model
        
        elif model_type == "prophet":
            # Load model
            model_path = os.path.join(self.model_dir, f"prophet_{symbol}.pkl")
            with open(model_path, "rb") as f:
                model = joblib.load(f)
            
            # Store model
            self.models[symbol] = model
            
            return model
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def predict_lstm(
        self,
        data: pd.DataFrame,
        symbol: str,
        target_col: str = "price",
        steps: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Make predictions with LSTM model.
        
        Args:
            data: DataFrame with time series data
            symbol: Symbol to make predictions for
            target_col: Column to predict
            steps: Number of steps to predict (default: forecast_horizon)
            
        Returns:
            DataFrame with predictions
        """
        steps = steps or self.forecast_horizon
        
        # Get model and scaler
        if symbol not in self.models:
            self.load_model(symbol, model_type="lstm")
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        
        # Prepare input data
        scaled_data = scaler.transform(data[[target_col]])
        
        # Use last lookback points as input
        input_data = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
        
        # Make predictions
        predictions = []
        current_input = input_data.copy()
        
        for _ in range(steps):
            # Predict next value
            next_pred = model.predict(current_input)[0, 0]
            predictions.append(next_pred)
            
            # Update input for next prediction
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, 0] = next_pred
        
        # Convert predictions to numpy array
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions)
        
        # Create DataFrame with predictions
        last_date = pd.to_datetime(data.index[-1])
        dates = [last_date + timedelta(days=i+1) for i in range(steps)]
        
        result = pd.DataFrame({
            "timestamp": dates,
            "prediction": predictions.flatten()
        })
        
        return result
    
    def predict_prophet(
        self,
        data: pd.DataFrame,
        symbol: str,
        date_col: str = "timestamp",
        steps: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Make predictions with Prophet model.
        
        Args:
            data: DataFrame with time series data
            symbol: Symbol to make predictions for
            date_col: Column with dates
            steps: Number of steps to predict (default: forecast_horizon)
            
        Returns:
            DataFrame with predictions
        """
        steps = steps or self.forecast_horizon
        
        # Get model
        if symbol not in self.models:
            self.load_model(symbol, model_type="prophet")
        
        model = self.models[symbol]
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=steps)
        
        # Make predictions
        forecast = model.predict(future)
        
        # Extract predictions for future period
        future_forecast = forecast.iloc[-steps:]
        
        # Create DataFrame with predictions
        result = pd.DataFrame({
            "timestamp": future_forecast["ds"],
            "prediction": future_forecast["yhat"],
            "prediction_lower": future_forecast["yhat_lower"],
            "prediction_upper": future_forecast["yhat_upper"]
        })
        
        return result
    
    def predict(
        self,
        data: pd.DataFrame,
        symbol: str,
        target_col: str = "price",
        date_col: str = "timestamp",
        model_type: Optional[str] = None,
        steps: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Make predictions.
        
        Args:
            data: DataFrame with time series data
            symbol: Symbol to make predictions for
            target_col: Column to predict
            date_col: Column with dates
            model_type: Type of model to use (lstm, prophet)
            steps: Number of steps to predict (default: forecast_horizon)
            
        Returns:
            DataFrame with predictions
        """
        # Use specified model type or default
        model_type = model_type or self.model_type
        
        if model_type == "lstm":
            return self.predict_lstm(data, symbol, target_col=target_col, steps=steps)
        elif model_type == "prophet":
            return self.predict_prophet(data, symbol, date_col=date_col, steps=steps)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample data
    dates = [datetime.now() - timedelta(days=i) for i in range(365, 0, -1)]
    prices = [100 + 10 * np.sin(i / 30) + i / 100 + np.random.normal(0, 1) for i in range(365)]
    
    data = pd.DataFrame({
        "timestamp": dates,
        "price": prices
    })
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster(model_dir="models", lookback=60, forecast_horizon=30)
    
    # Train LSTM model
    lstm_results = forecaster.train(data, "AAPL", model_type="lstm", epochs=10)
    
    # Make predictions
    predictions = forecaster.predict(data, "AAPL", model_type="lstm")
    
    print(predictions)

