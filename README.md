# E-commerce RAG Pipeline

A production-ready, industry-level multimodal Retrieval-Augmented Generation (RAG) pipeline for e-commerce product search and recommendations.

## 🌟 Features

### 🚀 Core Functionality
- **Multimodal Search**: Text and image queries using CLIP embeddings
- **Advanced Fusion**: Hybrid embedding strategies combining CLIP + SentenceTransformer
- **Fast Vector Search**: FAISS-powered similarity search with multiple index types
- **LLM Integration**: Support for Hugging Face, OpenAI, and Anthropic models
- **Field Extraction**: Smart extraction of product attributes (price, dimensions, etc.)
- **RESTful API**: FastAPI-based service with automatic documentation
- **CLI Interface**: Comprehensive command-line tools for all operations

### 🧠 Machine Learning
- **CLIP Fine-tuning**: Domain adaptation for e-commerce data
- **Contrastive Learning**: Custom loss functions for better embeddings
- **Embedding Fusion**: Multiple fusion strategies (weighted, concatenate, attention)
- **Model Training**: Automated training pipelines with Weights & Biases integration
- **Comprehensive Evaluation**: BLEU, ROUGE, recall, and latency metrics

### 🔧 Production Ready
- **Docker Support**: Multi-stage builds with GPU support
- **CI/CD Pipelines**: GitHub Actions for testing, training, and deployment
- **Model Hub Integration**: Automatic model sync to Hugging Face Hub
- **Monitoring**: Logging, metrics, and health checks
- **Testing**: Comprehensive test suite with coverage

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Input    │    │   Embedding      │    │   Vector DB     │
│   (CSV/API)     │───▶│   Service        │───▶│   (FAISS)       │
│                 │    │   (CLIP)         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐             │
│   Generated     │◀───│   LLM Client     │◀────────────┤
│   Response      │    │ (HF/OpenAI)      │             │
└─────────────────┘    └──────────────────┘             │
                                ▲                       │
                                │                       │
                       ┌─────────────────┐              │
                       │ Prompt Builder  │              │
                       │ & RAG Pipeline  │◀─────────────┘
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 8GB+ RAM (for CLIP model)
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd ecommerce_rag_pipeline
./scripts/setup.sh
```

2. **Activate environment**:
```bash
source venv/bin/activate
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Quick Demo

Run the demo to see the pipeline in action:

```bash
python scripts/demo.py
```

### Basic Usage

1. **Process your data**:
```bash
python -m src.cli process-data -i /path/to/your/products.csv -o data/
```

2. **Build search index**:
```bash
python -m src.cli build-index -i data/processed_data.csv -o models/
```

3. **Start API server**:
```bash
python -m src.cli serve --host 0.0.0.0 --port 8000
```

4. **Test search**:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"text_query": "wireless headphones", "k": 5}'
```

## 📊 Data Format

Your CSV file should contain these columns:

| Column | Description | Required |
|--------|-------------|----------|
| Product Name | Product title | ✅ |
| Category | Product category | ✅ |
| Selling Price | Price | ✅ |
| About Product | Description | ✅ |
| Image | Image URL | ✅ |
| Product Url | Product page URL | ✅ |
| Is Amazon Seller | Y/N flag | ✅ |
| Model Number | Model/SKU | ❌ |
| Product Specification | Technical specs | ❌ |
| Technical Details | Additional details | ❌ |
| Shipping Weight | Weight | ❌ |
| Product Dimensions | Dimensions | ❌ |
| Variants | Product variants | ❌ |

## 🔧 Configuration

Key environment variables:

```bash
# LLM Configuration
LLM_PROVIDER=anthropic  # anthropic, huggingface, openai
LLM_MODEL_NAME=claude-3-sonnet-20240229
LLM_API_TOKEN=your_api_token_here

# CLIP Model
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
DEVICE=cuda  # cuda, cpu, auto

# FAISS Index
FAISS_INDEX_TYPE=flat  # flat, ivf, hnsw
FAISS_METRIC=cosine  # cosine, l2, ip

# Model Training
HF_TOKEN=your_huggingface_token  # for model hub integration
WANDB_API_KEY=your_wandb_key     # optional, for experiment tracking
WANDB_PROJECT=ecommerce-rag-pipeline

# Paths
DATA_CSV_PATH=data/amazon_com_ecommerce.csv
FAISS_INDEX_PATH=models/product_index.faiss
FAISS_METADATA_PATH=models/product_metadata.pkl
```

## 🛠️ CLI Commands

### Data Processing
```bash
# Process raw CSV data
python -m src.cli process-data -i data.csv -o processed/

# Build search index
python -m src.cli build-index -i processed/data.csv -o models/

# Check pipeline status
python -m src.cli status
```

### Search and Query
```bash
# Search products
python -m src.cli search -q "wireless headphones" -k 5 --rerank

# Search with different output formats
python -m src.cli search -q "gaming laptop" --output-format json
python -m src.cli search -q "smartphone" --output-format detailed
```

### Model Training
```bash
# Train CLIP model on e-commerce data
python -m src.cli train-clip \
  --data-path data/amazon_com_ecommerce.csv \
  --epochs 5 \
  --batch-size 16 \
  --output-dir models/fine_tuned \
  --use-wandb

# Train embedding fusion model
python -m src.cli train-fusion \
  --data-path data/amazon_com_ecommerce.csv \
  --epochs 5 \
  --output-dir models/fusion \
  --use-wandb \
  --clip-model openai/clip-vit-base-patch32 \
  --sentence-model all-MiniLM-L6-v2

# Evaluate trained models
python -m src.cli evaluate-model \
  --model-path models/fine_tuned \
  --test-data data/test_set.csv \
  --output-file evaluation_results.json

# Push models to Hugging Face Hub
python -m src.cli push-to-hub \
  --model-path models/fine_tuned \
  --repo-name your-username/ecommerce-rag-clip \
  --private
```

### Evaluation and Deployment
```bash
# Run evaluation
python -m src.cli evaluate -i test_queries.json -o results.json

# Start API server
python -m src.cli serve --host 0.0.0.0 --port 8000 --reload
```

## 🌐 API Endpoints

### Search Endpoints

- `POST /search` - Text/multimodal search
- `POST /search/image` - Image upload search
- `POST /batch/search` - Batch search requests

### Utility Endpoints

- `GET /health` - Health check
- `GET /stats` - Pipeline statistics
- `POST /embeddings/text` - Generate text embeddings
- `POST /embeddings/image` - Generate image embeddings

### Example API Usage

```python
import requests

# Text search
response = requests.post("http://localhost:8000/search", json={
    "text_query": "wireless noise-canceling headphones",
    "k": 5,
    "rerank": True
})

# Image search
with open("product_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/search/image",
        files={"file": f},
        data={"k": 3}
    )

# Get embeddings
response = requests.post("http://localhost:8000/embeddings/text", json={
    "text": "Sony wireless headphones"
})
```

## 🐳 Docker Deployment

### Basic Deployment
```bash
# Build and run
docker-compose up --build

# Production deployment with scaling
docker-compose --profile production up -d
```

### Custom Configuration
```bash
# Override environment variables
docker run -e LLM_API_TOKEN=your_token \
           -e DATA_CSV_PATH=/data/products.csv \
           -v /host/data:/app/data \
           -p 8000:8000 \
           ecommerce-rag-pipeline
```

## ⚙️ CI/CD Pipelines

The project includes comprehensive GitHub Actions workflows for automated testing, training, and deployment:

### 🔄 Continuous Integration (`.github/workflows/ci.yml`)
- **Multi-Python Testing**: Tests across Python 3.9, 3.10, and 3.11
- **Code Quality**: Automated linting with flake8, formatting with black, import sorting with isort
- **Type Checking**: Static analysis with mypy
- **Security Scanning**: Vulnerability checks with bandit and safety
- **Docker Builds**: Automated container builds and deployments
- **Coverage Reporting**: Integration with Codecov

### 🤖 Model Training Pipeline (`.github/workflows/model-training.yml`)
- **Automated Training**: Triggered manually or on data updates
- **Weights & Biases Integration**: Experiment tracking and monitoring
- **Model Evaluation**: Automated performance assessment
- **Artifact Storage**: Model and evaluation result storage
- **Hugging Face Upload**: Automatic model publishing

### 🔄 Model Sync (`.github/workflows/model-sync.yml`)
- **Scheduled Sync**: Daily automatic model uploads to Hugging Face Hub
- **Model Validation**: Automated testing of uploaded models
- **Registry Management**: Maintains model registry with metadata
- **Notification**: Slack alerts for sync status

### 🚀 Release Pipeline (`.github/workflows/release.yml`)
- **PyPI Publishing**: Automated package releases
- **Docker Registry**: Multi-platform container builds
- **Documentation**: Automatic docs generation and deployment
- **GitHub Releases**: Automated release notes and assets

### Required GitHub Secrets
```bash
# Hugging Face Integration
HF_TOKEN=your_huggingface_token

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key

# Docker Registry
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password

# PyPI Publishing
PYPI_API_TOKEN=your_pypi_token
TEST_PYPI_API_TOKEN=your_test_pypi_token

# Notifications
SLACK_WEBHOOK_URL=your_slack_webhook
```

### Triggering Workflows
```bash
# Trigger model training
gh workflow run model-training.yml \
  -f model_type=clip_finetuning \
  -f data_path=data/amazon_com_ecommerce.csv \
  -f epochs=5 \
  -f use_wandb=true

# Trigger model sync
gh workflow run model-sync.yml \
  -f model_path=models/fine_tuned \
  -f repo_name=ecommerce-rag-clip \
  -f model_type=clip

# Create release
git tag v1.0.0
git push origin v1.0.0
```

## 🧪 Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test categories
pytest tests/test_data_processor.py -v
pytest tests/test_api.py -v
pytest tests/test_rag_pipeline.py -v

# Run with coverage
pytest --cov=src --cov-report=html tests/
```

## 📈 Evaluation

The pipeline includes comprehensive evaluation metrics:

### Retrieval Metrics
- **Recall@K**: Measures retrieval accuracy
- **Latency**: Search response time
- **Memory Usage**: Resource consumption

### Generation Metrics
- **BLEU Score**: N-gram overlap with reference
- **ROUGE-L**: Longest common subsequence
- **Human Evaluation**: Manual quality assessment

### Example Evaluation
```python
from src.rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator(pipeline)
results = evaluator.evaluate_end_to_end(
    test_queries=queries,
    reference_responses=references,
    k=5
)
```

## 🔍 Performance Tuning

### Index Optimization
```python
# For speed (large memory)
FAISS_INDEX_TYPE=flat

# For memory efficiency
FAISS_INDEX_TYPE=ivf
FAISS_NLIST=100

# For very large datasets
FAISS_INDEX_TYPE=hnsw
```

### Model Selection
```python
# Faster embedding (lower quality)
CLIP_MODEL_NAME=openai/clip-vit-base-patch16

# Better embedding (slower)
CLIP_MODEL_NAME=openai/clip-vit-large-patch14
```

### Batch Processing
```python
# Larger batches for throughput
BATCH_SIZE=64

# Smaller batches for memory constraints
BATCH_SIZE=16
```

## 📝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/
isort src/ tests/

# Run type checking
mypy src/

# Run linting
flake8 src/ tests/
```

## 📋 Roadmap

- [ ] **Advanced Reranking**: Cross-encoder models
- [ ] **Caching**: Redis-based result caching
- [ ] **Multi-language**: Support for multiple languages
- [ ] **Real-time Updates**: Dynamic index updates
- [ ] **A/B Testing**: Built-in experimentation framework
- [ ] **Advanced Analytics**: User behavior tracking
- [ ] **Federated Search**: Multi-source data integration

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use CPU instead
DEVICE=cpu

# Or use smaller batch size
BATCH_SIZE=16
```

**2. Model Download Fails**
```bash
# Set up Hugging Face cache
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

**3. API Token Issues**
```bash
# Verify token
python -c "from huggingface_hub import HfApi; HfApi().whoami(token='your_token')"
```

**4. Index Loading Fails**
```bash
# Rebuild index
python -m src.cli build-index -i data/processed_data.csv --force
```

### Performance Issues

**Slow Embedding Generation**
- Use GPU: `DEVICE=cuda`
- Increase batch size: `BATCH_SIZE=64`
- Use smaller model: `CLIP_MODEL_NAME=openai/clip-vit-base-patch16`

**High Memory Usage**
- Use IVF index: `FAISS_INDEX_TYPE=ivf`
- Reduce batch size: `BATCH_SIZE=16`
- Enable model offloading

**API Timeouts**
- Increase timeout in configuration
- Use async processing for large batches
- Implement request queuing

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) (when server is running)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [Development Guide](docs/development.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP** for multimodal embeddings
- **Facebook FAISS** for efficient similarity search
- **Hugging Face** for model hosting and APIs
- **FastAPI** for the web framework
- **All contributors** who helped improve this project

## 📧 Support

- Create an issue: [GitHub Issues](https://github.com/your-repo/issues)
- Documentation: [Project Wiki](https://github.com/your-repo/wiki)
- Email: your.email@example.com

---

**Built with ❤️ for the e-commerce community**