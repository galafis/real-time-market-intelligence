# 🇧🇷 Plataforma de Inteligência de Mercado em Tempo Real | 🇺🇸 Real-Time Market Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-000?style=for-the-badge&logo=apachekafka)
![WebSocket](https://img.shields.io/badge/WebSocket-010101?style=for-the-badge&logo=socketdotio&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![ClickHouse](https://img.shields.io/badge/ClickHouse-FFCC01?style=for-the-badge&logo=clickhouse&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Plataforma enterprise de inteligência de mercado financeiro com processamento de dados em tempo real, análise preditiva e dashboard interativo para traders e analistas**

[📊 Dashboard](#-dashboard-interativo) • [⚡ Real-Time](#-streaming-em-tempo-real) • [🤖 AI Models](#-modelos-de-ia) • [📈 Analytics](#-analytics-avançado)

</div>

---

## 🇧🇷 Português

### 🎯 Visão Geral

Plataforma **enterprise-grade** de inteligência de mercado que combina múltiplas fontes de dados financeiros para fornecer insights em tempo real:

- 📊 **Múltiplas Fontes**: Alpha Vantage, Yahoo Finance, News APIs, Social Media
- ⚡ **Processamento Real-Time**: Apache Kafka + WebSockets + ClickHouse
- 🤖 **Machine Learning**: Previsão de preços, análise de sentimento, detecção de anomalias
- 📈 **Analytics Avançado**: Indicadores técnicos, correlações, volatilidade
- 🎨 **Dashboard Interativo**: React + D3.js com visualizações em tempo real
- 🔔 **Sistema de Alertas**: Notificações inteligentes baseadas em ML

### 🏆 Objetivos da Plataforma

- **Processar dados** de 10,000+ ativos em tempo real
- **Prever movimentos** de preços com precisão >75%
- **Detectar anomalias** de mercado em <5 segundos
- **Analisar sentimento** de notícias e redes sociais
- **Fornecer insights** acionáveis para traders e analistas

### 🛠️ Stack Tecnológico Avançado

#### Data Ingestion & Streaming
- **Python 3.9+**: Linguagem principal para data engineering
- **Apache Kafka**: Message streaming para dados financeiros
- **WebSockets**: Conexões real-time com exchanges
- **Celery**: Task queue para processamento assíncrono
- **Redis**: Cache e message broker
- **Apache Airflow**: Orquestração de workflows

#### Data Storage & Processing
- **ClickHouse**: OLAP database para time-series
- **PostgreSQL**: Database transacional
- **MinIO**: Object storage para dados históricos
- **Apache Spark**: Processamento distribuído
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica

#### Machine Learning & AI
- **TensorFlow**: Deep learning para previsões
- **Scikit-learn**: Algoritmos clássicos de ML
- **Prophet**: Forecasting de séries temporais
- **NLTK/spaCy**: Processamento de linguagem natural
- **Transformers**: Modelos de sentimento avançados
- **MLflow**: ML lifecycle management

#### Frontend & Visualization
- **React 18**: Frontend framework
- **TypeScript**: Type-safe JavaScript
- **D3.js**: Visualizações interativas
- **Chart.js**: Gráficos financeiros
- **Material-UI**: Component library
- **Socket.IO**: Real-time communication

#### APIs & Integration
- **FastAPI**: High-performance API framework
- **Alpha Vantage API**: Dados de mercado
- **Yahoo Finance API**: Dados históricos
- **News API**: Notícias financeiras
- **Twitter API**: Análise de sentimento social
- **WebSocket APIs**: Dados real-time

#### DevOps & Monitoring
- **Docker**: Containerização
- **Kubernetes**: Orquestração
- **Prometheus**: Monitoring
- **Grafana**: Visualização de métricas
- **ELK Stack**: Logging e analytics
- **Terraform**: Infrastructure as Code

### 📋 Arquitetura da Plataforma

```
real-time-market-intelligence/
├── 📁 backend/                       # Backend Python
│   ├── 📁 src/                       # Código fonte principal
│   │   ├── 📁 data_ingestion/        # Ingestão de dados
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 alpha_vantage_client.py # Cliente Alpha Vantage
│   │   │   ├── 📄 yahoo_finance_client.py # Cliente Yahoo Finance
│   │   │   ├── 📄 news_api_client.py # Cliente News API
│   │   │   ├── 📄 twitter_client.py  # Cliente Twitter API
│   │   │   ├── 📄 websocket_client.py # Cliente WebSocket
│   │   │   ├── 📄 data_validator.py  # Validação de dados
│   │   │   └── 📄 ingestion_manager.py # Gerenciador de ingestão
│   │   ├── 📁 data_processing/       # Processamento de dados
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 technical_indicators.py # Indicadores técnicos
│   │   │   ├── 📄 price_calculator.py # Calculadora de preços
│   │   │   ├── 📄 volatility_analyzer.py # Análise de volatilidade
│   │   │   ├── 📄 correlation_analyzer.py # Análise de correlação
│   │   │   ├── 📄 anomaly_detector.py # Detector de anomalias
│   │   │   └── 📄 data_enricher.py   # Enriquecimento de dados
│   │   ├── 📁 ml_models/             # Modelos de Machine Learning
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 price_predictor.py # Preditor de preços
│   │   │   ├── 📄 sentiment_analyzer.py # Analisador de sentimento
│   │   │   ├── 📄 trend_detector.py  # Detector de tendências
│   │   │   ├── 📄 volatility_predictor.py # Preditor de volatilidade
│   │   │   ├── 📄 news_impact_analyzer.py # Análise impacto notícias
│   │   │   └── 📄 ensemble_predictor.py # Ensemble de modelos
│   │   ├── 📁 api/                   # API REST
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 main.py            # FastAPI app principal
│   │   │   ├── 📄 routers/           # Routers da API
│   │   │   │   ├── 📄 __init__.py    # Inicialização
│   │   │   │   ├── 📄 market_data.py # Endpoints dados mercado
│   │   │   │   ├── 📄 predictions.py # Endpoints predições
│   │   │   │   ├── 📄 analytics.py   # Endpoints analytics
│   │   │   │   ├── 📄 alerts.py      # Endpoints alertas
│   │   │   │   └── 📄 websocket.py   # WebSocket endpoints
│   │   │   ├── 📄 schemas/           # Pydantic schemas
│   │   │   │   ├── 📄 __init__.py    # Inicialização
│   │   │   │   ├── 📄 market_data.py # Schema dados mercado
│   │   │   │   ├── 📄 predictions.py # Schema predições
│   │   │   │   └── 📄 analytics.py   # Schema analytics
│   │   │   └── 📄 middleware/        # Middlewares
│   │   │       ├── 📄 __init__.py    # Inicialização
│   │   │       ├── 📄 auth.py        # Autenticação
│   │   │       ├── 📄 rate_limit.py  # Rate limiting
│   │   │       └── 📄 cors.py        # CORS middleware
│   │   ├── 📁 streaming/             # Processamento streaming
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 kafka_producer.py  # Producer Kafka
│   │   │   ├── 📄 kafka_consumer.py  # Consumer Kafka
│   │   │   ├── 📄 stream_processor.py # Processador streams
│   │   │   ├── 📄 websocket_server.py # Servidor WebSocket
│   │   │   └── 📄 real_time_analyzer.py # Análise tempo real
│   │   ├── 📁 database/              # Database e ORM
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 models.py          # Modelos SQLAlchemy
│   │   │   ├── 📄 clickhouse_client.py # Cliente ClickHouse
│   │   │   ├── 📄 postgres_client.py # Cliente PostgreSQL
│   │   │   ├── 📄 redis_client.py    # Cliente Redis
│   │   │   └── 📄 migrations/        # Migrações database
│   │   ├── 📁 alerts/                # Sistema de alertas
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 alert_engine.py    # Engine de alertas
│   │   │   ├── 📄 rule_engine.py     # Engine de regras
│   │   │   ├── 📄 notification_service.py # Serviço notificações
│   │   │   └── 📄 alert_templates.py # Templates de alertas
│   │   ├── 📁 utils/                 # Utilitários
│   │   │   ├── 📄 __init__.py        # Inicialização
│   │   │   ├── 📄 config.py          # Configurações
│   │   │   ├── 📄 logger.py          # Logger customizado
│   │   │   ├── 📄 cache.py           # Cache utilities
│   │   │   ├── 📄 validators.py      # Validadores
│   │   │   └── 📄 helpers.py         # Funções auxiliares
│   │   └── 📁 monitoring/            # Monitoramento
│   │       ├── 📄 __init__.py        # Inicialização
│   │       ├── 📄 metrics_collector.py # Coleta métricas
│   │       ├── 📄 health_checker.py  # Health checks
│   │       ├── 📄 performance_monitor.py # Monitor performance
│   │       └── 📄 dashboard_metrics.py # Métricas dashboard
│   ├── 📁 tests/                     # Testes automatizados
│   │   ├── 📁 unit/                  # Testes unitários
│   │   ├── 📁 integration/           # Testes integração
│   │   ├── 📁 performance/           # Testes performance
│   │   └── 📁 data/                  # Dados para testes
│   ├── 📄 requirements.txt           # Dependências Python
│   ├── 📄 requirements-dev.txt       # Dependências desenvolvimento
│   ├── 📄 Dockerfile                # Docker backend
│   └── 📄 docker-compose.yml         # Docker compose
├── 📁 frontend/                      # Frontend React
│   ├── 📁 public/                    # Arquivos públicos
│   │   ├── 📄 index.html             # HTML principal
│   │   ├── 📄 manifest.json          # PWA manifest
│   │   └── 📄 favicon.ico            # Favicon
│   ├── 📁 src/                       # Código fonte React
│   │   ├── 📁 components/            # Componentes React
│   │   │   ├── 📁 common/            # Componentes comuns
│   │   │   │   ├── 📄 Header.tsx     # Header da aplicação
│   │   │   │   ├── 📄 Sidebar.tsx    # Sidebar navegação
│   │   │   │   ├── 📄 LoadingSpinner.tsx # Loading spinner
│   │   │   │   └── 📄 ErrorBoundary.tsx # Error boundary
│   │   │   ├── 📁 charts/            # Componentes gráficos
│   │   │   │   ├── 📄 CandlestickChart.tsx # Gráfico candlestick
│   │   │   │   ├── 📄 LineChart.tsx  # Gráfico linha
│   │   │   │   ├── 📄 VolumeChart.tsx # Gráfico volume
│   │   │   │   ├── 📄 HeatMap.tsx    # Mapa de calor
│   │   │   │   └── 📄 TechnicalIndicators.tsx # Indicadores
│   │   │   ├── 📁 dashboard/         # Componentes dashboard
│   │   │   │   ├── 📄 MarketOverview.tsx # Visão geral mercado
│   │   │   │   ├── 📄 WatchList.tsx  # Lista observação
│   │   │   │   ├── 📄 NewsPanel.tsx  # Painel notícias
│   │   │   │   ├── 📄 AlertsPanel.tsx # Painel alertas
│   │   │   │   └── 📄 PredictionsPanel.tsx # Painel predições
│   │   │   ├── 📁 analytics/         # Componentes analytics
│   │   │   │   ├── 📄 CorrelationMatrix.tsx # Matriz correlação
│   │   │   │   ├── 📄 VolatilityAnalysis.tsx # Análise volatilidade
│   │   │   │   ├── 📄 SentimentAnalysis.tsx # Análise sentimento
│   │   │   │   └── 📄 TrendAnalysis.tsx # Análise tendências
│   │   │   └── 📁 trading/           # Componentes trading
│   │   │       ├── 📄 OrderBook.tsx  # Livro de ofertas
│   │   │       ├── 📄 TradeHistory.tsx # Histórico trades
│   │   │       ├── 📄 PositionManager.tsx # Gerenciador posições
│   │   │       └── 📄 RiskMetrics.tsx # Métricas risco
│   │   ├── 📁 pages/                 # Páginas da aplicação
│   │   │   ├── 📄 Dashboard.tsx      # Dashboard principal
│   │   │   ├── 📄 Analytics.tsx      # Página analytics
│   │   │   ├── 📄 Trading.tsx        # Página trading
│   │   │   ├── 📄 Alerts.tsx         # Página alertas
│   │   │   └── 📄 Settings.tsx       # Página configurações
│   │   ├── 📁 hooks/                 # Custom hooks
│   │   │   ├── 📄 useWebSocket.ts    # Hook WebSocket
│   │   │   ├── 📄 useMarketData.ts   # Hook dados mercado
│   │   │   ├── 📄 usePredictions.ts  # Hook predições
│   │   │   └── 📄 useAlerts.ts       # Hook alertas
│   │   ├── 📁 services/              # Serviços API
│   │   │   ├── 📄 api.ts             # Cliente API
│   │   │   ├── 📄 websocket.ts       # Serviço WebSocket
│   │   │   ├── 📄 marketData.ts      # Serviço dados mercado
│   │   │   └── 📄 predictions.ts     # Serviço predições
│   │   ├── 📁 store/                 # Estado global (Redux)
│   │   │   ├── 📄 index.ts           # Store principal
│   │   │   ├── 📄 marketSlice.ts     # Slice dados mercado
│   │   │   ├── 📄 predictionsSlice.ts # Slice predições
│   │   │   └── 📄 alertsSlice.ts     # Slice alertas
│   │   ├── 📁 utils/                 # Utilitários frontend
│   │   │   ├── 📄 formatters.ts      # Formatadores
│   │   │   ├── 📄 calculations.ts    # Cálculos
│   │   │   └── 📄 constants.ts       # Constantes
│   │   ├── 📁 types/                 # Tipos TypeScript
│   │   │   ├── 📄 market.ts          # Tipos dados mercado
│   │   │   ├── 📄 predictions.ts     # Tipos predições
│   │   │   └── 📄 alerts.ts          # Tipos alertas
│   │   ├── 📄 App.tsx                # Componente principal
│   │   ├── 📄 index.tsx              # Entry point
│   │   └── 📄 index.css              # Estilos globais
│   ├── 📄 package.json               # Dependências Node.js
│   ├── 📄 tsconfig.json              # Configuração TypeScript
│   ├── 📄 Dockerfile                # Docker frontend
│   └── 📄 .env.example               # Exemplo variáveis ambiente
├── 📁 data/                          # Dados e datasets
│   ├── 📁 raw/                       # Dados brutos
│   │   ├── 📄 market_data_sample.csv # Amostra dados mercado
│   │   ├── 📄 news_data_sample.json  # Amostra notícias
│   │   └── 📄 social_sentiment.csv   # Sentimento redes sociais
│   ├── 📁 processed/                 # Dados processados
│   │   ├── 📄 technical_indicators.parquet # Indicadores técnicos
│   │   ├── 📄 sentiment_scores.parquet # Scores sentimento
│   │   └── 📄 predictions_history.parquet # Histórico predições
│   └── 📁 models/                    # Modelos treinados
│       ├── 📄 price_predictor_v1.pkl # Preditor preços
│       ├── 📄 sentiment_model_v1.pkl # Modelo sentimento
│       └── 📄 anomaly_detector_v1.pkl # Detector anomalias
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 📄 01_data_exploration.ipynb  # Exploração dados
│   ├── 📄 02_technical_analysis.ipynb # Análise técnica
│   ├── 📄 03_sentiment_analysis.ipynb # Análise sentimento
│   ├── 📄 04_price_prediction.ipynb  # Predição preços
│   ├── 📄 05_anomaly_detection.ipynb # Detecção anomalias
│   └── 📄 06_backtesting.ipynb       # Backtesting estratégias
├── 📁 config/                        # Configurações
│   ├── 📄 app_config.yaml           # Configuração aplicação
│   ├── 📄 kafka_config.yaml         # Configuração Kafka
│   ├── 📄 database_config.yaml      # Configuração databases
│   ├── 📄 api_keys.yaml.example     # Exemplo chaves API
│   └── 📄 monitoring_config.yaml    # Configuração monitoramento
├── 📁 deployment/                    # Deployment e infraestrutura
│   ├── 📁 docker/                   # Docker configs
│   │   ├── 📄 docker-compose.prod.yml # Docker compose produção
│   │   ├── 📄 docker-compose.dev.yml # Docker compose desenvolvimento
│   │   └── 📄 nginx.conf             # Configuração Nginx
│   ├── 📁 kubernetes/               # Kubernetes manifests
│   │   ├── 📄 namespace.yaml        # Namespace
│   │   ├── 📄 backend-deployment.yaml # Backend deployment
│   │   ├── 📄 frontend-deployment.yaml # Frontend deployment
│   │   ├── 📄 kafka-deployment.yaml # Kafka deployment
│   │   ├── 📄 clickhouse-deployment.yaml # ClickHouse deployment
│   │   └── 📄 ingress.yaml          # Ingress
│   └── 📁 terraform/                # Infrastructure as Code
│       ├── 📄 main.tf               # Main Terraform config
│       ├── 📄 variables.tf          # Variables
│       ├── 📄 outputs.tf            # Outputs
│       └── 📄 modules/              # Terraform modules
├── 📁 scripts/                       # Scripts utilitários
│   ├── 📄 setup_environment.sh      # Setup ambiente
│   ├── 📄 start_services.sh         # Iniciar serviços
│   ├── 📄 data_ingestion.py         # Script ingestão dados
│   ├── 📄 model_training.py         # Treinamento modelos
│   └── 📄 deployment.sh             # Script deployment
├── 📁 docs/                          # Documentação
│   ├── 📄 README.md                 # Este arquivo
│   ├── 📄 ARCHITECTURE.md           # Documentação arquitetura
│   ├── 📄 API_REFERENCE.md          # Referência API
│   ├── 📄 DEPLOYMENT_GUIDE.md       # Guia deployment
│   ├── 📄 USER_GUIDE.md             # Guia usuário
│   └── 📁 images/                   # Imagens documentação
├── 📄 .gitignore                    # Arquivos ignorados
├── 📄 LICENSE                       # Licença MIT
├── 📄 Makefile                      # Comandos make
├── 📄 .env.example                  # Exemplo variáveis ambiente
└── 📄 .github/                      # GitHub workflows
    └── 📄 workflows/                # CI/CD workflows
        ├── 📄 ci.yml                # Continuous Integration
        ├── 📄 cd.yml                # Continuous Deployment
        └── 📄 data-pipeline.yml     # Pipeline dados
```

### 📊 Dashboard Interativo

#### 1. 🎨 Interface Principal

**Componentes do Dashboard**
```typescript
interface MarketDashboard {
  // Visão geral do mercado
  marketOverview: {
    majorIndices: MarketIndex[];
    topMovers: Stock[];
    sectorPerformance: SectorData[];
    marketSentiment: SentimentScore;
  };
  
  // Watchlist personalizada
  watchList: {
    stocks: WatchedStock[];
    alerts: Alert[];
    customIndicators: TechnicalIndicator[];
  };
  
  // Análise em tempo real
  realTimeAnalysis: {
    priceMovements: PriceData[];
    volumeAnalysis: VolumeData[];
    newsImpact: NewsImpact[];
    socialSentiment: SocialSentiment[];
  };
  
  // Predições ML
  predictions: {
    priceForecasts: PriceForecast[];
    trendPredictions: TrendPrediction[];
    volatilityForecasts: VolatilityForecast[];
    riskAssessments: RiskAssessment[];
  };
}
```

#### 2. 📈 Gráficos Avançados

**Candlestick Chart com Indicadores**
```typescript
const CandlestickChart: React.FC<CandlestickProps> = ({
  data,
  indicators,
  timeframe,
  onTimeframeChange
}) => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>([]);
  
  // WebSocket para dados real-time
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/market-data');
    
    ws.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setChartData(prev => [...prev.slice(-1000), newData]);
    };
    
    return () => ws.close();
  }, []);
  
  // Configuração D3.js para gráfico
  const chartConfig = {
    width: 800,
    height: 400,
    margin: { top: 20, right: 30, bottom: 40, left: 50 },
    indicators: {
      sma: { period: 20, color: '#ff6b6b' },
      ema: { period: 12, color: '#4ecdc4' },
      bollinger: { period: 20, stdDev: 2, color: '#45b7d1' },
      rsi: { period: 14, overbought: 70, oversold: 30 },
      macd: { fast: 12, slow: 26, signal: 9 }
    }
  };
  
  return (
    <div className="candlestick-chart">
      <div className="chart-controls">
        <TimeframeSelector 
          value={timeframe}
          onChange={onTimeframeChange}
          options={['1m', '5m', '15m', '1h', '4h', '1d']}
        />
        <IndicatorSelector
          selected={selectedIndicators}
          onChange={setSelectedIndicators}
          available={Object.keys(chartConfig.indicators)}
        />
      </div>
      
      <svg ref={chartRef} width={chartConfig.width} height={chartConfig.height}>
        {/* D3.js rendering logic */}
      </svg>
      
      <div className="chart-legend">
        {selectedIndicators.map(indicator => (
          <LegendItem key={indicator} indicator={indicator} />
        ))}
      </div>
    </div>
  );
};
```

### ⚡ Streaming em Tempo Real

#### 1. 🔄 Kafka Producer para Dados de Mercado

**Producer Otimizado para Alta Frequência**
```python
class MarketDataProducer:
    def __init__(self, config: Dict):
        self.config = config
        self.producer = KafkaProducer(
            bootstrap_servers=config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8'),
            acks='1',  # Balance between performance and reliability
            retries=3,
            batch_size=32768,  # Larger batch for high throughput
            linger_ms=5,  # Low latency
            compression_type='lz4',  # Fast compression
            max_in_flight_requests_per_connection=10
        )
        
        # Initialize API clients
        self.alpha_vantage = AlphaVantageClient(config['api_keys']['alpha_vantage'])
        self.yahoo_finance = YahooFinanceClient()
        self.news_api = NewsAPIClient(config['api_keys']['news_api'])
        
    async def stream_market_data(self, symbols: List[str]):
        """Stream real-time market data for given symbols."""
        
        tasks = []
        for symbol in symbols:
            # Create tasks for different data sources
            tasks.extend([
                self._stream_price_data(symbol),
                self._stream_volume_data(symbol),
                self._stream_news_data(symbol),
                self._stream_social_sentiment(symbol)
            ])
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
    
    async def _stream_price_data(self, symbol: str):
        """Stream price data for a symbol."""
        
        while True:
            try:
                # Get real-time price data
                price_data = await self.alpha_vantage.get_real_time_price(symbol)
                
                # Enrich with technical indicators
                enriched_data = await self._calculate_technical_indicators(
                    symbol, price_data
                )
                
                # Send to Kafka
                self.producer.send(
                    topic='market-data-prices',
                    key=symbol,
                    value={
                        'symbol': symbol,
                        'timestamp': datetime.utcnow().isoformat(),
                        'data_type': 'price',
                        'data': enriched_data
                    }
                )
                
                await asyncio.sleep(1)  # 1-second intervals
                
            except Exception as e:
                logger.error(f"Error streaming price data for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _calculate_technical_indicators(self, symbol: str, price_data: Dict) -> Dict:
        """Calculate technical indicators for price data."""
        
        # Get historical data for calculations
        historical_data = await self._get_historical_data(symbol, periods=100)
        
        if not historical_data:
            return price_data
        
        # Calculate indicators
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_20'] = self._calculate_sma(historical_data, 20)
        indicators['sma_50'] = self._calculate_sma(historical_data, 50)
        
        # Exponential Moving Averages
        indicators['ema_12'] = self._calculate_ema(historical_data, 12)
        indicators['ema_26'] = self._calculate_ema(historical_data, 26)
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(historical_data, 14)
        
        # MACD
        macd_data = self._calculate_macd(historical_data)
        indicators.update(macd_data)
        
        # Bollinger Bands
        bollinger_data = self._calculate_bollinger_bands(historical_data, 20, 2)
        indicators.update(bollinger_data)
        
        # Volume indicators
        indicators['volume_sma'] = self._calculate_volume_sma(historical_data, 20)
        indicators['volume_ratio'] = price_data.get('volume', 0) / indicators['volume_sma']
        
        return {
            **price_data,
            'technical_indicators': indicators
        }
    
    def _calculate_sma(self, data: List[Dict], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return None
        
        prices = [float(d['close']) for d in data[-period:]]
        return sum(prices) / len(prices)
    
    def _calculate_ema(self, data: List[Dict], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return None
        
        prices = [float(d['close']) for d in data]
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        ema = sum(prices[:period]) / period
        
        # Calculate EMA for remaining periods
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, data: List[Dict], period: int) -> float:
        """Calculate Relative Strength Index."""
        if len(data) < period + 1:
            return None
        
        prices = [float(d['close']) for d in data]
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
```

#### 2. 📊 WebSocket Server para Frontend

**WebSocket Server com Múltiplos Canais**
```python
class MarketDataWebSocketServer:
    def __init__(self, kafka_config: Dict):
        self.kafka_config = kafka_config
        self.connections: Dict[str, Set[WebSocket]] = {
            'market_data': set(),
            'predictions': set(),
            'alerts': set(),
            'news': set()
        }
        
        # Kafka consumer for real-time data
        self.consumer = KafkaConsumer(
            'market-data-prices',
            'market-data-news',
            'market-predictions',
            'market-alerts',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='websocket-server',
            enable_auto_commit=True
        )
    
    async def websocket_endpoint(self, websocket: WebSocket, channel: str):
        """Handle WebSocket connections for different channels."""
        
        await websocket.accept()
        
        if channel not in self.connections:
            await websocket.close(code=4000, reason="Invalid channel")
            return
        
        self.connections[channel].add(websocket)
        logger.info(f"Client connected to channel: {channel}")
        
        try:
            # Send initial data
            await self._send_initial_data(websocket, channel)
            
            # Keep connection alive and handle client messages
            while True:
                try:
                    # Wait for client message with timeout
                    message = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=30.0
                    )
                    
                    # Handle client message (subscription updates, etc.)
                    await self._handle_client_message(websocket, channel, message)
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_text(json.dumps({
                        'type': 'ping',
                        'timestamp': datetime.utcnow().isoformat()
                    }))
                    
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from channel: {channel}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connections[channel].discard(websocket)
    
    async def _send_initial_data(self, websocket: WebSocket, channel: str):
        """Send initial data when client connects."""
        
        if channel == 'market_data':
            # Send current market overview
            market_overview = await self._get_market_overview()
            await websocket.send_text(json.dumps({
                'type': 'market_overview',
                'data': market_overview,
                'timestamp': datetime.utcnow().isoformat()
            }))
            
        elif channel == 'predictions':
            # Send latest predictions
            predictions = await self._get_latest_predictions()
            await websocket.send_text(json.dumps({
                'type': 'predictions',
                'data': predictions,
                'timestamp': datetime.utcnow().isoformat()
            }))
    
    async def start_kafka_consumer(self):
        """Start consuming Kafka messages and broadcast to WebSocket clients."""
        
        async for message in self.consumer:
            try:
                topic = message.topic
                data = message.value
                
                # Determine which channel to broadcast to
                if topic == 'market-data-prices':
                    await self._broadcast_to_channel('market_data', {
                        'type': 'price_update',
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                elif topic == 'market-predictions':
                    await self._broadcast_to_channel('predictions', {
                        'type': 'prediction_update',
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                elif topic == 'market-alerts':
                    await self._broadcast_to_channel('alerts', {
                        'type': 'alert',
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                elif topic == 'market-data-news':
                    await self._broadcast_to_channel('news', {
                        'type': 'news_update',
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
    
    async def _broadcast_to_channel(self, channel: str, message: Dict):
        """Broadcast message to all clients in a channel."""
        
        if channel not in self.connections:
            return
        
        disconnected_clients = set()
        message_text = json.dumps(message)
        
        for websocket in self.connections[channel]:
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected_clients.add(websocket)
        
        # Remove disconnected clients
        self.connections[channel] -= disconnected_clients
```

### 🤖 Modelos de IA

#### 1. 📈 Preditor de Preços com LSTM

**Deep Learning para Previsão de Preços**
```python
class PricePredictionModel:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower'
        ]
        
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model for price prediction."""
        
        model = Sequential([
            # First LSTM layer with dropout
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            # Dense layers
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')  # Price prediction
        ])
        
        # Compile with custom loss function
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Select and scale features
        features = df[self.feature_columns].values
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(scaled_features[i, 3])  # Close price index
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """Train the price prediction model."""
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                'best_price_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history.history
        }
    
    def predict(self, recent_data: pd.DataFrame, steps_ahead: int = 1) -> List[float]:
        """Make price predictions."""
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare input data
        features = recent_data[self.feature_columns].values
        scaled_features = self.scaler.transform(features)
        
        predictions = []
        current_sequence = scaled_features[-60:]  # Last 60 time steps
        
        for _ in range(steps_ahead):
            # Reshape for prediction
            X = current_sequence.reshape(1, 60, len(self.feature_columns))
            
            # Make prediction
            pred_scaled = self.model.predict(X, verbose=0)[0, 0]
            
            # Inverse transform to get actual price
            # Create dummy array for inverse transform
            dummy_features = np.zeros((1, len(self.feature_columns)))
            dummy_features[0, 3] = pred_scaled  # Close price index
            pred_actual = self.scaler.inverse_transform(dummy_features)[0, 3]
            
            predictions.append(pred_actual)
            
            # Update sequence for next prediction
            # This is simplified - in practice, you'd need to predict all features
            new_row = current_sequence[-1].copy()
            new_row[3] = pred_scaled  # Update close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return predictions
```

### 🎯 Competências Demonstradas

#### Data Engineering & Streaming
- ✅ **Apache Kafka**: Streaming de dados financeiros em alta frequência
- ✅ **WebSockets**: Comunicação real-time com frontend
- ✅ **ClickHouse**: OLAP database para time-series
- ✅ **Data Pipelines**: ETL/ELT para dados financeiros
- ✅ **API Integration**: Múltiplas fontes de dados (Alpha Vantage, Yahoo, News)

#### Machine Learning & AI
- ✅ **Deep Learning**: LSTM para previsão de preços
- ✅ **Time Series Forecasting**: Prophet e modelos customizados
- ✅ **Sentiment Analysis**: NLP para análise de notícias e redes sociais
- ✅ **Anomaly Detection**: Detecção de movimentos anômalos de mercado
- ✅ **Feature Engineering**: Indicadores técnicos automatizados

#### Frontend & Visualization
- ✅ **React + TypeScript**: Frontend moderno e type-safe
- ✅ **D3.js**: Visualizações financeiras interativas
- ✅ **Real-time Updates**: WebSocket integration
- ✅ **Responsive Design**: Interface adaptável
- ✅ **State Management**: Redux para estado global

---

## 🇺🇸 English

### 🎯 Overview

**Enterprise-grade** market intelligence platform that combines multiple financial data sources to provide real-time insights:

- 📊 **Multiple Sources**: Alpha Vantage, Yahoo Finance, News APIs, Social Media
- ⚡ **Real-Time Processing**: Apache Kafka + WebSockets + ClickHouse
- 🤖 **Machine Learning**: Price prediction, sentiment analysis, anomaly detection
- 📈 **Advanced Analytics**: Technical indicators, correlations, volatility
- 🎨 **Interactive Dashboard**: React + D3.js with real-time visualizations
- 🔔 **Alert System**: ML-based intelligent notifications

### 🏆 Platform Objectives

- **Process data** from 10,000+ assets in real-time
- **Predict price movements** with >75% accuracy
- **Detect market anomalies** in <5 seconds
- **Analyze sentiment** from news and social media
- **Provide actionable insights** for traders and analysts

### 🎯 Skills Demonstrated

#### Data Engineering & Streaming
- ✅ **Apache Kafka**: High-frequency financial data streaming
- ✅ **WebSockets**: Real-time frontend communication
- ✅ **ClickHouse**: OLAP database for time-series
- ✅ **Data Pipelines**: ETL/ELT for financial data
- ✅ **API Integration**: Multiple data sources integration

#### Machine Learning & AI
- ✅ **Deep Learning**: LSTM for price prediction
- ✅ **Time Series Forecasting**: Prophet and custom models
- ✅ **Sentiment Analysis**: NLP for news and social media
- ✅ **Anomaly Detection**: Abnormal market movement detection
- ✅ **Feature Engineering**: Automated technical indicators

#### Frontend & Visualization
- ✅ **React + TypeScript**: Modern type-safe frontend
- ✅ **D3.js**: Interactive financial visualizations
- ✅ **Real-time Updates**: WebSocket integration
- ✅ **Responsive Design**: Adaptive interface
- ✅ **State Management**: Redux for global state

---

## 📄 Licença | License

MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes | see [LICENSE](LICENSE) file for details

## 📞 Contato | Contact

**GitHub**: [@galafis](https://github.com/galafis)  
**LinkedIn**: [Gabriel Demetrios Lafis](https://linkedin.com/in/galafis)  
**Email**: gabriel.lafis@example.com

---

<div align="center">

**Desenvolvido com ❤️ para Inteligência de Mercado | Developed with ❤️ for Market Intelligence**

[![GitHub](https://img.shields.io/badge/GitHub-galafis-blue?style=flat-square&logo=github)](https://github.com/galafis)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)](https://reactjs.org/)

</div>

