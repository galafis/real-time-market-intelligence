# 🇧🇷 Plataforma de Inteligência de Mercado em Tempo Real

![Imagem Hero da Plataforma de Inteligência de Mercado em Tempo Real](./hero_image.png)

[![Status do Projeto](https://img.shields.io/badge/Status-Ativo-brightgreen)](https://github.com/galafis/real-time-market-intelligence)
[![Licença](https://img.shields.io/badge/Licença-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Kafka-3.4.0-red)](https://kafka.apache.org/)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-23.3-yellow)](https://clickhouse.com/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB)](https://reactjs.org/)

## 📊 Visão Geral

A **Plataforma de Inteligência de Mercado em Tempo Real** é um sistema enterprise-grade para análise e visualização de dados financeiros em tempo real. A plataforma integra dados de múltiplas fontes (APIs financeiras, feeds de notícias, redes sociais) e fornece insights acionáveis através de análises avançadas e visualizações interativas.

### 🌟 Características Principais

- **Processamento em Tempo Real**: Ingestão e análise de dados com latência <100ms usando Apache Kafka e ClickHouse
- **Múltiplas Fontes de Dados**: Integração com Alpha Vantage, Yahoo Finance, Twitter, Bloomberg e outras APIs financeiras
- **Análise Preditiva**: Modelos LSTM e Prophet para previsão de tendências de mercado
- **Análise de Sentimento**: Processamento de notícias e mídias sociais para análise de sentimento de mercado
- **Dashboard Interativo**: Visualizações avançadas com React, TypeScript e D3.js
- **Escalabilidade**: Arquitetura distribuída capaz de processar 10.000+ ativos simultaneamente
- **Alta Disponibilidade**: Tolerância a falhas e recuperação automática

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python 3.9+**: Linguagem principal para processamento de dados e APIs
- **Apache Kafka**: Streaming de dados em tempo real
- **ClickHouse**: Armazenamento analítico colunar para consultas de alta performance
- **FastAPI**: API RESTful de alto desempenho
- **Redis**: Cache e pub/sub para dados em tempo real
- **Pandas & NumPy**: Manipulação e análise de dados
- **scikit-learn & TensorFlow**: Modelos de machine learning e deep learning

### Frontend
- **React 18**: Biblioteca JavaScript para construção de interfaces
- **TypeScript**: Tipagem estática para JavaScript
- **D3.js**: Visualizações de dados avançadas e interativas
- **Material-UI**: Componentes de interface de usuário
- **Redux**: Gerenciamento de estado da aplicação
- **WebSockets**: Comunicação bidirecional em tempo real

### DevOps
- **Docker & Docker Compose**: Containerização e orquestração
- **Kubernetes**: Orquestração de contêineres para produção
- **Prometheus & Grafana**: Monitoramento e alertas
- **GitHub Actions**: CI/CD automatizado

## 🏗️ Arquitetura

A plataforma segue uma arquitetura de microsserviços orientada a eventos:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│  Data Ingestion │────▶│  Kafka Cluster  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Web Dashboard  │◀────│   API Gateway   │◀────│  Stream Proc.   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │   ClickHouse    │
                                               │    Cluster      │
                                               └─────────────────┘
```

### Componentes Principais:

1. **Data Ingestion Service**: Coleta dados de APIs financeiras, feeds de notícias e redes sociais
2. **Kafka Cluster**: Backbone de streaming para processamento de eventos em tempo real
3. **Stream Processing**: Análise em tempo real, detecção de anomalias e enriquecimento de dados
4. **ClickHouse Cluster**: Armazenamento analítico para consultas de alta performance
5. **API Gateway**: Endpoints RESTful e WebSockets para acesso aos dados
6. **Web Dashboard**: Interface de usuário interativa para visualização e análise

## 🚀 Instalação e Uso

### Pré-requisitos

- Docker e Docker Compose
- Python 3.9+
- Node.js 16+
- Chaves de API para serviços financeiros (Alpha Vantage, Yahoo Finance, etc.)

### Configuração do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/galafis/real-time-market-intelligence.git
cd real-time-market-intelligence
```

2. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas chaves de API e configurações
```

3. Inicie os serviços com Docker Compose:
```bash
docker-compose up -d
```

4. Inicialize o banco de dados:
```bash
python src/scripts/initialize_db.py
```

5. Inicie a aplicação web:
```bash
cd frontend
npm install
npm start
```

### Uso Básico

1. Acesse o dashboard em `http://localhost:3000`
2. Configure os ativos financeiros que deseja monitorar
3. Visualize dados em tempo real, análises e previsões
4. Configure alertas para condições específicas de mercado

## 📊 Exemplos de Uso

### Monitoramento de Ativos em Tempo Real

```python
from src.client import MarketIntelligenceClient

# Inicializar cliente
client = MarketIntelligenceClient(api_key="sua_chave_api")

# Monitorar ativos em tempo real
client.subscribe_to_assets(["AAPL", "MSFT", "GOOGL"], callback=process_update)

# Função de callback para processar atualizações
def process_update(update):
    print(f"Ativo: {update["symbol"]}, Preço: {update["price"]}, Variação: {update["change_percent"]}% ")
```

### Análise de Sentimento de Mercado

```python
# Analisar sentimento de mercado para um ativo
sentiment = client.get_market_sentiment("AAPL")

print(f"Sentimento: {sentiment["score"]}")
print(f"Fontes positivas: {sentiment["positive_sources"]}")
print(f"Fontes negativas: {sentiment["negative_sources"]}")
```

### Previsão de Tendências

```python
# Obter previsão para os próximos 7 dias
forecast = client.get_price_forecast("AAPL", days=7)

for date, prediction in forecast.items():
    print(f"Data: {date}, Preço previsto: {prediction["price"]}, Intervalo de confiança: {prediction["confidence_interval"]}")
```

## 📁 Estrutura do Projeto

```
real-time-market-intelligence/
├── src/
│   ├── api/                # API RESTful e WebSockets
│   ├── data/               # Ingestão e processamento de dados
│   ├── models/             # Modelos de ML para previsão e análise
│   ├── streaming/          # Processamento de streaming com Kafka
│   ├── visualization/      # Componentes de visualização
│   ├── config/             # Configurações da aplicação
│   └── utils/              # Utilitários e helpers
├── frontend/               # Aplicação React para dashboard
├── tests/                  # Testes unitários e de integração
├── docs/                   # Documentação
├── docker/                 # Arquivos Docker e configurações
├── scripts/                # Scripts de utilidade
└── notebooks/              # Jupyter notebooks para análises
```

## 📈 Casos de Uso

1. **Trading Algorítmico**: Fornece dados em tempo real e sinais para sistemas de trading automatizados
2. **Análise de Portfólio**: Monitoramento e análise de desempenho de portfólios de investimento
3. **Detecção de Anomalias**: Identificação de movimentos anormais de mercado e oportunidades
4. **Análise de Sentimento**: Correlação entre notícias, mídias sociais e movimentos de mercado
5. **Backtesting de Estratégias**: Teste de estratégias de investimento com dados históricos

## 🔧 Desenvolvimento

### Configuração do Ambiente de Desenvolvimento

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements-dev.txt

# Configurar hooks de pre-commit
pre-commit install
```

### Executando Testes

```bash
# Executar todos os testes
pytest

# Executar testes com cobertura
pytest --cov=src tests/
```

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Contribuições

Contribuições são bem-vindas! Por favor, sinta-se à vontade para enviar um Pull Request.

---

# 🇬🇧 Real-Time Market Intelligence Platform

![Hero Image for Real-Time Market Intelligence Platform](./hero_image.png)

[![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/galafis/real-time-market-intelligence)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Kafka-3.4.0-red)](https://kafka.apache.org/)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-23.3-yellow)](https://clickhouse.com/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB)](https://reactjs.org/)

## 📊 Overview

The **Real-Time Market Intelligence Platform** is an enterprise-grade system for real-time financial data analysis and visualization. The platform integrates data from multiple sources (financial APIs, news feeds, social media) and provides actionable insights through advanced analytics and interactive visualizations.

### 🌟 Key Features

- **Real-Time Processing**: Data ingestion and analysis with <100ms latency using Apache Kafka and ClickHouse
- **Multiple Data Sources**: Integration with Alpha Vantage, Yahoo Finance, Twitter, Bloomberg, and other financial APIs
- **Predictive Analytics**: LSTM and Prophet models for market trend forecasting
- **Sentiment Analysis**: Processing of news and social media for market sentiment analysis
- **Interactive Dashboard**: Advanced visualizations with React, TypeScript, and D3.js
- **Scalability**: Distributed architecture capable of processing 10,000+ assets simultaneously
- **High Availability**: Fault tolerance and automatic recovery

## 🛠️ Technologies Used

### Backend
- **Python 3.9+**: Main language for data processing and APIs
- **Apache Kafka**: Real-time data streaming
- **ClickHouse**: Columnar analytical storage for high-performance queries
- **FastAPI**: High-performance RESTful API
- **Redis**: Cache and pub/sub for real-time data
- **Pandas & NumPy**: Data manipulation and analysis
- **scikit-learn & TensorFlow**: Machine learning and deep learning

### Frontend
- **React 18**: JavaScript library for building interfaces
- **TypeScript**: Static typing for JavaScript
- **D3.js**: Advanced and interactive data visualizations
- **Material-UI**: User interface components
- **Redux**: Application state management
- **WebSockets**: Real-time bidirectional communication

### DevOps
- **Docker & Docker Compose**: Containerization and orchestration
- **Kubernetes**: Container orchestration for production
- **Prometheus & Grafana**: Monitoramento e alertas
- **GitHub Actions**: Automated CI/CD

## 🏗️ Architecture

The platform follows an event-driven microservices architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│  Data Ingestion │────▶│  Kafka Cluster  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Web Dashboard  │◀────│   API Gateway   │◀────│  Stream Proc.   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │   ClickHouse    │
                                               │    Cluster      │
                                               └─────────────────┘
```

### Main Components:

1. **Data Ingestion Service**: Collects data from financial APIs, news feeds, and social media
2. **Kafka Cluster**: Streaming backbone for real-time event processing
3. **Stream Processing**: Real-time analysis, anomaly detection, and data enrichment
4. **ClickHouse Cluster**: Analytical storage for high-performance queries
5. **API Gateway**: RESTful and WebSocket endpoints for data access
6. **Web Dashboard**: Interactive user interface for visualization and analysis

## 🚀 Installation and Usage

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Node.js 16+
- API keys for financial services (Alpha Vantage, Yahoo Finance, etc.)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/galafis/real-time-market-intelligence.git
cd real-time-market-intelligence
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit the .env file with your API keys and configurations
```

3. Start services with Docker Compose:
```bash
docker-compose up -d
```

4. Initialize the database:
```bash
python src/scripts/initialize_db.py
```

5. Start the web application:
```bash
cd frontend
npm install
npm start
```

### Basic Usage

1. Access the dashboard at `http://localhost:3000`
2. Configure the financial assets you want to monitor
3. View real-time data, analytics, and forecasts
4. Set up alerts for specific market conditions

## 📊 Usage Examples

### Real-Time Asset Monitoring

```python
from src.client import MarketIntelligenceClient

# Initialize client
client = MarketIntelligenceClient(api_key="your_api_key")

# Monitor assets in real-time
client.subscribe_to_assets(["AAPL", "MSFT", "GOOGL"], callback=process_update)

# Callback function to process updates
def process_update(update):
    print(f"Asset: {update["symbol"]}, Price: {update["price"]}, Change: {update["change_percent"]}% ")
```

### Market Sentiment Analysis

```python
# Analyze market sentiment for an asset
sentiment = client.get_market_sentiment("AAPL")

print(f"Sentiment: {sentiment["score"]}")
print(f"Positive sources: {sentiment["positive_sources"]}")
print(f"Negative sources: {sentiment["negative_sources"]}")
```

### Trend Forecasting

```python
# Get forecast for the next 7 days
forecast = client.get_price_forecast("AAPL", days=7)

for date, prediction in forecast.items():
    print(f"Date: {date}, Predicted price: {prediction["price"]}, Confidence interval: {prediction["confidence_interval"]}")
```

## 📁 Project Structure

```
real-time-market-intelligence/
├── src/
│   ├── api/                # RESTful API and WebSockets
│   ├── data/               # Data ingestion and processing
│   ├── models/             # ML models for prediction and analysis
│   ├── streaming/          # Streaming processing with Kafka
│   ├── visualization/      # Visualization components
│   ├── config/             # Application configurations
│   └── utils/              # Utilities and helpers
├── frontend/               # React application for dashboard
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── docker/                 # Docker files and configurations
├── scripts/                # Utility scripts
└── notebooks/              # Jupyter notebooks for analysis
```

## 📈 Use Cases

1. **Algorithmic Trading**: Provides real-time data and signals for automated trading systems
2. **Portfolio Analysis**: Monitoring and analysis of investment portfolio performance
3. **Anomaly Detection**: Identification of abnormal market movements and opportunities
4. **Sentiment Analysis**: Correlation between news, social media, and market movements
5. **Backtesting de Estratégias**: Testing investment strategies with historical data

## 🔧 Development

### Development Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributions

Contributions are welcome! Please feel free to submit a Pull Request.


