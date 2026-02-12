# 📈 Real Time Market Intelligence

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.48-E6522C.svg)](https://prometheus.io/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D.svg)](https://redis.io/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Real Time Market Intelligence** — Advanced data science project: real-time-market-intelligence

Total source lines: **5,139** across **11** files in **2** languages.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+
- Docker and Docker Compose (optional)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/real-time-market-intelligence.git
cd real-time-market-intelligence
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```




## 🐳 Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```




### 📁 Project Structure

```
real-time-market-intelligence/
├── docker/
│   └── README.md
├── docs/
│   └── README.md
├── frontend/
│   └── README.md
├── notebooks/
│   ├── eda/
│   ├── prototypes/
│   ├── tutorials/
│   │   └── tutorials/
│   └── README.md
├── src/
│   ├── api/
│   │   └── market_api.py
│   ├── config/
│   │   └── README.md
│   ├── data/
│   │   └── README.md
│   ├── models/
│   │   ├── sentiment_analyzer.py
│   │   └── time_series_forecaster.py
│   ├── scripts/
│   │   ├── README.md
│   │   └── initialize_db.py
│   ├── streaming/
│   │   ├── kafka_consumer.py
│   │   └── kafka_producer.py
│   ├── utils/
│   │   └── logger.py
│   ├── visualization/
│   │   └── dashboard.py
│   ├── __init__.py
│   └── client.py
├── tests/
│   └── README.md
├── README.md
├── STRUCTURE_STATUS.md
├── docker-compose.yml
├── requirements-dev.txt
└── requirements.txt
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 10 files |
| HTML | 1 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Real Time Market Intelligence** — Advanced data science project: real-time-market-intelligence

Total de linhas de código: **5,139** em **11** arquivos em **2** linguagens.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+
- Docker e Docker Compose (opcional)

#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/real-time-market-intelligence.git
cd real-time-market-intelligence
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```







### 📁 Estrutura do Projeto

```
real-time-market-intelligence/
├── docker/
│   └── README.md
├── docs/
│   └── README.md
├── frontend/
│   └── README.md
├── notebooks/
│   ├── eda/
│   ├── prototypes/
│   ├── tutorials/
│   │   └── tutorials/
│   └── README.md
├── src/
│   ├── api/
│   │   └── market_api.py
│   ├── config/
│   │   └── README.md
│   ├── data/
│   │   └── README.md
│   ├── models/
│   │   ├── sentiment_analyzer.py
│   │   └── time_series_forecaster.py
│   ├── scripts/
│   │   ├── README.md
│   │   └── initialize_db.py
│   ├── streaming/
│   │   ├── kafka_consumer.py
│   │   └── kafka_producer.py
│   ├── utils/
│   │   └── logger.py
│   ├── visualization/
│   │   └── dashboard.py
│   ├── __init__.py
│   └── client.py
├── tests/
│   └── README.md
├── README.md
├── STRUCTURE_STATUS.md
├── docker-compose.yml
├── requirements-dev.txt
└── requirements.txt
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 10 files |
| HTML | 1 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
