"""
Real-Time Market Intelligence Platform
Sentiment Analyzer Module

This module provides functionality to analyze sentiment in financial news
and social media content using NLP techniques.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import re
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import threading
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Try to import transformers for advanced models
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer for financial news and social media content.
    
    This class provides methods to analyze sentiment in financial text
    using both traditional ML and transformer-based approaches.
    """
    
    def __init__(
        self,
        model_type: str = "ensemble",
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        download_nltk: bool = True
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type: Type of model to use ("traditional", "transformer", "ensemble")
            model_path: Path to pre-trained model (if None, use default)
            use_gpu: Whether to use GPU for transformer models
            download_nltk: Whether to download NLTK resources
        """
        self.model_type = model_type
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False
        
        # Download NLTK resources if needed
        if download_nltk:
            self._download_nltk_resources()
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize models
        self.traditional_model = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        # Load models
        self._load_models()
        
        # Financial lexicon for domain-specific sentiment
        self.financial_lexicon = self._load_financial_lexicon()
        
        logger.info(f"Initialized SentimentAnalyzer with model type: {model_type}")
    
    def _download_nltk_resources(self) -> None:
        """Download required NLTK resources."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("Downloaded NLTK resources")
        except Exception as e:
            logger.warning(f"Failed to download NLTK resources: {str(e)}")
    
    def _load_financial_lexicon(self) -> Dict[str, float]:
        """
        Load financial sentiment lexicon.
        
        Returns:
            Dictionary mapping financial terms to sentiment scores
        """
        # Default financial lexicon (simplified)
        lexicon = {
            # Positive terms
            "growth": 0.8,
            "profit": 0.9,
            "increase": 0.7,
            "gain": 0.8,
            "positive": 0.7,
            "bullish": 0.9,
            "outperform": 0.8,
            "upgrade": 0.8,
            "beat": 0.7,
            "strong": 0.6,
            "opportunity": 0.6,
            "recovery": 0.7,
            "success": 0.8,
            "innovation": 0.7,
            "dividend": 0.6,
            
            # Negative terms
            "loss": -0.9,
            "decline": -0.7,
            "decrease": -0.7,
            "negative": -0.7,
            "bearish": -0.9,
            "underperform": -0.8,
            "downgrade": -0.8,
            "miss": -0.7,
            "weak": -0.6,
            "risk": -0.6,
            "debt": -0.6,
            "bankruptcy": -0.9,
            "lawsuit": -0.8,
            "investigation": -0.7,
            "volatility": -0.5
        }
        
        # Try to load custom lexicon if available
        custom_lexicon_path = os.path.join(
            os.path.dirname(__file__), 
            "../data/financial_sentiment_lexicon.json"
        )
        
        if os.path.exists(custom_lexicon_path):
            try:
                with open(custom_lexicon_path, 'r') as f:
                    custom_lexicon = json.load(f)
                lexicon.update(custom_lexicon)
                logger.info(f"Loaded custom financial lexicon with {len(custom_lexicon)} terms")
            except Exception as e:
                logger.error(f"Error loading custom financial lexicon: {str(e)}")
        
        return lexicon
    
    def _load_models(self) -> None:
        """Load sentiment analysis models."""
        # Load traditional model
        if self.model_type in ["traditional", "ensemble"]:
            self._load_traditional_model()
        
        # Load transformer model
        if self.model_type in ["transformer", "ensemble"]:
            self._load_transformer_model()
    
    def _load_traditional_model(self) -> None:
        """Load traditional ML model for sentiment analysis."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.traditional_model = pickle.load(f)
                logger.info(f"Loaded traditional model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading traditional model: {str(e)}")
                self._create_default_traditional_model()
        else:
            self._create_default_traditional_model()
    
    def _create_default_traditional_model(self) -> None:
        """Create a default traditional ML model for sentiment analysis."""
        logger.info("Creating default traditional sentiment model")
        
        # Create a simple pipeline with TF-IDF and Random Forest
        self.traditional_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
        
        # Note: This model needs to be trained before use
        logger.warning("Default traditional model created but not trained")
    
    def _load_transformer_model(self) -> None:
        """Load transformer model for sentiment analysis."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available, skipping transformer model")
            return
        
        try:
            # Load FinBERT model for financial sentiment analysis
            model_name = "ProsusAI/finbert"
            
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to GPU if available
            if self.use_gpu:
                self.transformer_model = self.transformer_model.to('cuda')
                logger.info("Transformer model loaded on GPU")
            else:
                logger.info("Transformer model loaded on CPU")
        
        except Exception as e:
            logger.error(f"Error loading transformer model: {str(e)}")
            self.transformer_model = None
            self.transformer_tokenizer = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def analyze_sentiment_lexicon(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using the financial lexicon.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = preprocessed_text.split()
        
        # Calculate sentiment score
        sentiment_score = 0.0
        matched_terms = []
        
        for token in tokens:
            if token in self.financial_lexicon:
                score = self.financial_lexicon[token]
                sentiment_score += score
                matched_terms.append((token, score))
        
        # Normalize score
        if matched_terms:
            sentiment_score /= len(matched_terms)
        
        # Determine sentiment label
        if sentiment_score > 0.1:
            sentiment_label = "positive"
        elif sentiment_score < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            "score": sentiment_score,
            "label": sentiment_label,
            "matched_terms": matched_terms,
            "confidence": abs(sentiment_score) if matched_terms else 0.0
        }
    
    def analyze_sentiment_traditional(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using the traditional ML model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.traditional_model is None:
            logger.error("Traditional model not loaded")
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "error": "Model not loaded"
            }
        
        try:
            # Preprocess text
            preprocessed_text = self.preprocess_text(text)
            
            # Check if model is trained
            if not hasattr(self.traditional_model, "classes_"):
                logger.error("Traditional model not trained")
                return {
                    "score": 0.0,
                    "label": "neutral",
                    "confidence": 0.0,
                    "error": "Model not trained"
                }
            
            # Predict sentiment
            prediction = self.traditional_model.predict([preprocessed_text])[0]
            probabilities = self.traditional_model.predict_proba([preprocessed_text])[0]
            
            # Map prediction to label
            if prediction == 1:
                sentiment_label = "positive"
                sentiment_score = probabilities[1]
            elif prediction == -1:
                sentiment_label = "negative"
                sentiment_score = -probabilities[0]
            else:
                sentiment_label = "neutral"
                sentiment_score = 0.0
            
            return {
                "score": sentiment_score,
                "label": sentiment_label,
                "confidence": max(probabilities)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment with traditional model: {str(e)}")
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_sentiment_transformer(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using the transformer model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not TRANSFORMERS_AVAILABLE or self.transformer_model is None:
            logger.error("Transformer model not available")
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "error": "Model not available"
            }
        
        try:
            # Tokenize text
            inputs = self.transformer_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to GPU if available
            if self.use_gpu:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()[0]
            
            # FinBERT labels: 0 = negative, 1 = neutral, 2 = positive
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            prediction = int(np.argmax(probabilities))
            
            # Calculate sentiment score (-1 to 1)
            if prediction == 2:  # Positive
                sentiment_score = probabilities[2]
            elif prediction == 0:  # Negative
                sentiment_score = -probabilities[0]
            else:  # Neutral
                sentiment_score = 0.0
            
            return {
                "score": float(sentiment_score),
                "label": label_map[prediction],
                "confidence": float(probabilities[prediction]),
                "probabilities": {
                    "negative": float(probabilities[0]),
                    "neutral": float(probabilities[1]),
                    "positive": float(probabilities[2])
                }
            }
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment with transformer model: {str(e)}")
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment in text using the configured model(s).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        results = {}
        
        # Always run lexicon-based analysis
        lexicon_results = self.analyze_sentiment_lexicon(text)
        results["lexicon"] = lexicon_results
        
        # Run traditional model if configured
        if self.model_type in ["traditional", "ensemble"]:
            traditional_results = self.analyze_sentiment_traditional(text)
            results["traditional"] = traditional_results
        
        # Run transformer model if configured
        if self.model_type in ["transformer", "ensemble"] and TRANSFORMERS_AVAILABLE:
            transformer_results = self.analyze_sentiment_transformer(text)
            results["transformer"] = transformer_results
        
        # Ensemble results if using ensemble model
        if self.model_type == "ensemble":
            ensemble_results = self._ensemble_results(results)
            results["ensemble"] = ensemble_results
            
            # Use ensemble as final result
            final_results = ensemble_results
        elif self.model_type == "transformer" and TRANSFORMERS_AVAILABLE:
            # Use transformer as final result
            final_results = results["transformer"]
        elif self.model_type == "traditional":
            # Use traditional as final result
            final_results = results["traditional"]
        else:
            # Fall back to lexicon
            final_results = results["lexicon"]
        
        # Add all results
        final_results["all_results"] = results
        
        return final_results
    
    def _ensemble_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensemble results from different models.
        
        Args:
            results: Dictionary with results from different models
            
        Returns:
            Dictionary with ensembled results
        """
        # Extract scores and confidences
        scores = []
        confidences = []
        
        if "lexicon" in results:
            scores.append(results["lexicon"]["score"])
            confidences.append(results["lexicon"]["confidence"])
        
        if "traditional" in results and "error" not in results["traditional"]:
            scores.append(results["traditional"]["score"])
            confidences.append(results["traditional"]["confidence"])
        
        if "transformer" in results and "error" not in results["transformer"]:
            scores.append(results["transformer"]["score"])
            confidences.append(results["transformer"]["confidence"])
        
        # Calculate weighted average score
        if confidences:
            total_confidence = sum(confidences)
            if total_confidence > 0:
                weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
            else:
                weighted_score = sum(scores) / len(scores) if scores else 0.0
        else:
            weighted_score = sum(scores) / len(scores) if scores else 0.0
        
        # Determine sentiment label
        if weighted_score > 0.1:
            sentiment_label = "positive"
        elif weighted_score < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        # Calculate overall confidence
        overall_confidence = max(confidences) if confidences else 0.0
        
        return {
            "score": weighted_score,
            "label": sentiment_label,
            "confidence": overall_confidence
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def train(
        self,
        texts: List[str],
        labels: List[int],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the traditional sentiment analysis model.
        
        Args:
            texts: List of texts for training
            labels: List of sentiment labels (1 for positive, 0 for neutral, -1 for negative)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results
        """
        if self.traditional_model is None:
            logger.error("Traditional model not initialized")
            return {"error": "Model not initialized"}
        
        try:
            # Preprocess texts
            preprocessed_texts = [self.preprocess_text(text) for text in texts]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                preprocessed_texts, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
            
            # Train model
            self.traditional_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.traditional_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Trained traditional model: accuracy={accuracy:.4f}, f1={f1:.4f}")
            
            return {
                "accuracy": accuracy,
                "f1_score": f1,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
        
        except Exception as e:
            logger.error(f"Error training traditional model: {str(e)}")
            return {"error": str(e)}
    
    def save_model(self, path: str) -> bool:
        """
        Save the traditional model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.traditional_model is None:
            logger.error("No traditional model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            with open(path, 'wb') as f:
                pickle.dump(self.traditional_model, f)
            
            logger.info(f"Saved traditional model to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving traditional model: {str(e)}")
            return False


class MarketSentimentAnalyzer:
    """
    Specialized sentiment analyzer for financial market data.
    
    This class extends the base SentimentAnalyzer with market-specific
    functionality, such as entity recognition and aspect-based sentiment.
    """
    
    def __init__(
        self,
        model_type: str = "ensemble",
        model_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the market sentiment analyzer.
        
        Args:
            model_type: Type of model to use ("traditional", "transformer", "ensemble")
            model_path: Path to pre-trained model (if None, use default)
            use_gpu: Whether to use GPU for transformer models
        """
        # Initialize base sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(
            model_type=model_type,
            model_path=model_path,
            use_gpu=use_gpu
        )
        
        # Financial entity recognition
        self.company_symbols = self._load_company_symbols()
        
        # Aspect categories for financial sentiment
        self.aspects = [
            "earnings", "revenue", "growth", "management",
            "products", "competition", "regulation", "market"
        ]
        
        logger.info("Initialized MarketSentimentAnalyzer")
    
    def _load_company_symbols(self) -> Dict[str, str]:
        """
        Load company symbols and names.
        
        Returns:
            Dictionary mapping company symbols to names
        """
        # Default company symbols (simplified)
        symbols = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "AMZN": "Amazon",
            "META": "Meta",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA",
            "JPM": "JPMorgan Chase",
            "BAC": "Bank of America",
            "WMT": "Walmart"
        }
        
        # Try to load custom symbols if available
        custom_symbols_path = os.path.join(
            os.path.dirname(__file__), 
            "../data/company_symbols.json"
        )
        
        if os.path.exists(custom_symbols_path):
            try:
                with open(custom_symbols_path, 'r') as f:
                    custom_symbols = json.load(f)
                symbols.update(custom_symbols)
                logger.info(f"Loaded custom company symbols: {len(custom_symbols)} companies")
            except Exception as e:
                logger.error(f"Error loading custom company symbols: {str(e)}")
        
        return symbols
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract financial entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of dictionaries with entity information
        """
        entities = []
        
        # Extract company symbols and names
        text_lower = text.lower()
        
        for symbol, name in self.company_symbols.items():
            # Check for symbol (case sensitive)
            if symbol in text or f"${symbol}" in text:
                entities.append({
                    "type": "company",
                    "symbol": symbol,
                    "name": name,
                    "match_type": "symbol"
                })
            # Check for company name (case insensitive)
            elif name.lower() in text_lower:
                entities.append({
                    "type": "company",
                    "symbol": symbol,
                    "name": name,
                    "match_type": "name"
                })
        
        # Extract aspects
        for aspect in self.aspects:
            if aspect.lower() in text_lower:
                entities.append({
                    "type": "aspect",
                    "name": aspect
                })
        
        return entities
    
    def analyze_market_sentiment(
        self,
        text: str,
        extract_entities: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze sentiment in financial market text.
        
        Args:
            text: Text to analyze
            extract_entities: Whether to extract entities
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Analyze general sentiment
        sentiment_results = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Extract entities if requested
        if extract_entities:
            entities = self.extract_entities(text)
            
            # Analyze entity-specific sentiment
            entity_sentiments = []
            
            for entity in entities:
                if entity["type"] == "company":
                    # Extract sentences mentioning the entity
                    entity_text = ""
                    
                    # Check for symbol
                    symbol = entity["symbol"]
                    if symbol in text or f"${symbol}" in text:
                        # Find sentences containing the symbol
                        sentences = re.split(r'[.!?]', text)
                        for sentence in sentences:
                            if symbol in sentence or f"${symbol}" in sentence:
                                entity_text += sentence + ". "
                    
                    # Check for name
                    name = entity["name"]
                    if name in text:
                        # Find sentences containing the name
                        sentences = re.split(r'[.!?]', text)
                        for sentence in sentences:
                            if name in sentence:
                                entity_text += sentence + ". "
                    
                    # Analyze sentiment for entity text
                    if entity_text:
                        entity_sentiment = self.sentiment_analyzer.analyze_sentiment(entity_text)
                        
                        entity_sentiments.append({
                            "entity": entity,
                            "sentiment": {
                                "score": entity_sentiment["score"],
                                "label": entity_sentiment["label"],
                                "confidence": entity_sentiment["confidence"]
                            },
                            "text": entity_text.strip()
                        })
            
            # Add entity sentiments to results
            sentiment_results["entities"] = entities
            sentiment_results["entity_sentiments"] = entity_sentiments
        
        return sentiment_results
    
    def analyze_news_batch(
        self,
        news_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of news items.
        
        Args:
            news_items: List of news items with 'headline' and 'content' fields
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        results = []
        
        for item in news_items:
            # Combine headline and content
            text = item.get('headline', '')
            if 'content' in item and item['content']:
                text += ". " + item['content']
            
            # Analyze sentiment
            sentiment = self.analyze_market_sentiment(text)
            
            # Add to results
            results.append({
                "news_item": item,
                "sentiment": sentiment
            })
        
        return results
    
    def analyze_social_batch(
        self,
        social_posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of social media posts.
        
        Args:
            social_posts: List of social media posts with 'text' field
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        results = []
        
        for post in social_posts:
            # Get text
            text = post.get('text', '')
            
            # Analyze sentiment
            sentiment = self.analyze_market_sentiment(text)
            
            # Add to results
            results.append({
                "post": post,
                "sentiment": sentiment
            })
        
        return results
    
    def generate_market_sentiment_report(
        self,
        symbol: str,
        news_items: List[Dict[str, Any]] = None,
        social_posts: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive market sentiment report for a symbol.
        
        Args:
            symbol: Stock symbol to analyze
            news_items: List of news items related to the symbol
            social_posts: List of social media posts related to the symbol
            
        Returns:
            Dictionary with market sentiment report
        """
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "news_sentiment": None,
            "social_sentiment": None,
            "overall_sentiment": None
        }
        
        # Analyze news sentiment
        if news_items:
            news_results = self.analyze_news_batch(news_items)
            
            # Calculate average sentiment
            sentiment_scores = [r["sentiment"]["score"] for r in news_results]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            # Determine sentiment label
            if avg_sentiment > 0.1:
                sentiment_label = "positive"
            elif avg_sentiment < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Add to report
            report["news_sentiment"] = {
                "score": avg_sentiment,
                "label": sentiment_label,
                "count": len(news_items),
                "positive_count": sum(1 for s in sentiment_scores if s > 0.1),
                "neutral_count": sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1),
                "negative_count": sum(1 for s in sentiment_scores if s < -0.1),
                "details": news_results
            }
        
        # Analyze social sentiment
        if social_posts:
            social_results = self.analyze_social_batch(social_posts)
            
            # Calculate average sentiment
            sentiment_scores = [r["sentiment"]["score"] for r in social_results]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            # Determine sentiment label
            if avg_sentiment > 0.1:
                sentiment_label = "positive"
            elif avg_sentiment < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Add to report
            report["social_sentiment"] = {
                "score": avg_sentiment,
                "label": sentiment_label,
                "count": len(social_posts),
                "positive_count": sum(1 for s in sentiment_scores if s > 0.1),
                "neutral_count": sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1),
                "negative_count": sum(1 for s in sentiment_scores if s < -0.1),
                "details": social_results
            }
        
        # Calculate overall sentiment
        news_score = report["news_sentiment"]["score"] if report["news_sentiment"] else 0.0
        social_score = report["social_sentiment"]["score"] if report["social_sentiment"] else 0.0
        
        # Weight news more heavily than social media
        if report["news_sentiment"] and report["social_sentiment"]:
            overall_score = (news_score * 0.7) + (social_score * 0.3)
        elif report["news_sentiment"]:
            overall_score = news_score
        elif report["social_sentiment"]:
            overall_score = social_score
        else:
            overall_score = 0.0
        
        # Determine overall sentiment label
        if overall_score > 0.1:
            overall_label = "positive"
        elif overall_score < -0.1:
            overall_label = "negative"
        else:
            overall_label = "neutral"
        
        # Add to report
        report["overall_sentiment"] = {
            "score": overall_score,
            "label": overall_label
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    analyzer = MarketSentimentAnalyzer(model_type="ensemble")
    
    # Example financial texts
    texts = [
        "Apple's quarterly earnings beat expectations, showing strong growth in services.",
        "Tesla stock plummets after disappointing delivery numbers and production delays.",
        "Microsoft announces new cloud partnership, but impact on revenue remains unclear.",
        "$AAPL is looking bullish with the new iPhone launch around the corner.",
        "Regulatory concerns continue to plague Facebook, with new investigations announced."
    ]
    
    # Analyze sentiment
    for text in texts:
        result = analyzer.analyze_market_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (score: {result['score']:.2f}, confidence: {result['confidence']:.2f})")
        
        if "entities" in result:
            print("Entities:")
            for entity in result["entities"]:
                if entity["type"] == "company":
                    print(f"  - Company: {entity['name']} ({entity['symbol']})")
                else:
                    print(f"  - Aspect: {entity['name']}")
        
        if "entity_sentiments" in result:
            print("Entity Sentiments:")
            for entity_sentiment in result["entity_sentiments"]:
                entity = entity_sentiment["entity"]
                sentiment = entity_sentiment["sentiment"]
                if entity["type"] == "company":
                    print(f"  - {entity['name']} ({entity['symbol']}): {sentiment['label']} (score: {sentiment['score']:.2f})")
        
        print()
    
    # Generate market sentiment report
    news_items = [
        {"headline": "Apple's quarterly earnings beat expectations", "content": "The company showed strong growth in services."},
        {"headline": "New iPhone expected to boost Apple sales", "content": "Analysts predict significant revenue increase."}
    ]
    
    social_posts = [
        {"text": "$AAPL is looking bullish with the new iPhone launch around the corner."},
        {"text": "Not impressed with the new Apple products. Overpriced as usual."}
    ]
    
    report = analyzer.generate_market_sentiment_report("AAPL", news_items, social_posts)
    
    print("Market Sentiment Report:")
    print(f"Symbol: {report['symbol']}")
    print(f"Overall Sentiment: {report['overall_sentiment']['label']} (score: {report['overall_sentiment']['score']:.2f})")
    
    if report["news_sentiment"]:
        print(f"News Sentiment: {report['news_sentiment']['label']} (score: {report['news_sentiment']['score']:.2f})")
        print(f"  - Positive: {report['news_sentiment']['positive_count']}")
        print(f"  - Neutral: {report['news_sentiment']['neutral_count']}")
        print(f"  - Negative: {report['news_sentiment']['negative_count']}")
    
    if report["social_sentiment"]:
        print(f"Social Sentiment: {report['social_sentiment']['label']} (score: {report['social_sentiment']['score']:.2f})")
        print(f"  - Positive: {report['social_sentiment']['positive_count']}")
        print(f"  - Neutral: {report['social_sentiment']['neutral_count']}")
        print(f"  - Negative: {report['social_sentiment']['negative_count']}")

