# E-commerce RAG Pipeline

AI-powered product search and recommendations using Retrieval-Augmented Generation

[![Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue)](https://huggingface.co/spaces/your-username/ecommerce-rag-pipeline)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/ycnliu/ecommerce-rag-pipeline)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This project implements a production-ready Retrieval-Augmented Generation (RAG) pipeline for e-commerce product search and recommendations. It combines semantic search using CLIP embeddings, efficient vector similarity search with FAISS, and AI-powered response generation using large language models.

**Key Capabilities:**
- Semantic product search that understands meaning, not just keywords
- Fast vector similarity search across thousands of products
- AI-generated personalized recommendations
- Multiple deployment options from local development to cloud hosting
- Automated CI/CD pipeline with GitHub Actions

## Architecture

The system follows a classic RAG architecture with the following components:

```
User Query → Embedding Generation → Vector Search → Context Retrieval → LLM Generation → Response
```

**Core Components:**

1. **Embedding Layer** (`src/embedding/`)
   - CLIP model for multimodal (text + image) embeddings
   - SentenceTransformers for advanced fusion strategies
   - Batch processing for efficient generation

2. **Vector Database** (`src/vector_db/`)
   - FAISS index supporting Flat, IVF, and HNSW algorithms
   - Configurable distance metrics (L2, Inner Product)
   - Metadata management with parallel storage

3. **RAG Pipeline** (`src/rag/`)
   - Query orchestration and context retrieval
   - Multiple LLM backend support (OpenAI, HuggingFace, Ollama)
   - Prompt engineering and response generation

4. **Data Processing** (`src/data/`)
   - CSV data loading and cleaning
   - Product metadata management
   - Pydantic models for validation

5. **Web Interface**
   - Gradio application for interactive demos
   - FastAPI server for REST endpoints

## Technology Stack

**Machine Learning:**
- OpenAI CLIP (openai/clip-vit-base-patch32) - 512-dimensional embeddings
- Mixtral-8x7B-Instruct - Text generation via HuggingFace Inference API
- SentenceTransformers (all-MiniLM-L6-v2) - Alternative embeddings
- FAISS - Vector similarity search

**Backend:**
- Python 3.9+
- FastAPI - REST API framework
- Gradio - Web demo interface
- Pydantic - Data validation
- NumPy/Pandas - Data processing

**DevOps:**
- Docker & Docker Compose
- GitHub Actions - CI/CD automation
- pytest - Testing framework
- black/isort/flake8 - Code quality tools

## Project Structure

```
ecommerce-rag-pipeline/
├── src/
│   ├── embedding/              # Embedding generation
│   │   ├── service.py         # CLIP embedding service
│   │   └── fusion.py          # Embedding fusion strategies
│   ├── vector_db/             # Vector database
│   │   ├── faiss_service.py  # FAISS index management
│   │   └── evaluation.py     # Performance metrics
│   ├── rag/                   # RAG pipeline
│   │   ├── rag_pipeline.py   # Main orchestration
│   │   ├── llm_client.py     # LLM integrations
│   │   └── prompt_builder.py # Prompt engineering
│   ├── data/                  # Data processing
│   │   ├── processor.py      # CSV processing
│   │   └── models.py         # Data models
│   ├── api/                   # REST API
│   │   └── main.py           # FastAPI application
│   └── utils/                 # Utilities
│       ├── config.py         # Configuration
│       └── exceptions.py     # Custom exceptions
│
├── data/                      # Datasets
│   ├── amazon_com_ecommerce.csv     # 10K products
│   └── test_sample_1000.csv         # Test subset
│
├── .github/workflows/         # CI/CD automation
│   ├── ci.yml                # Tests, linting, security
│   ├── model-training.yml    # Model fine-tuning
│   ├── model-sync.yml        # HuggingFace uploads
│   └── release.yml           # Package publishing
│
├── tests/                     # Test suite
│   ├── test_vector_db.py
│   ├── test_rag_pipeline.py
│   └── test_embedding.py
│
├── scripts/                   # Utility scripts
│   ├── create_model_card.py
│   └── check_hf_quota.py
│
├── app.py                     # Gradio web interface
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image
├── docker-compose.yml         # Multi-service setup
└── README.md                  # This file
```

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/ycnliu/ecommerce-rag-pipeline.git
cd ecommerce-rag-pipeline

# Install dependencies
pip install -r requirements.txt

# Install optional ML dependencies for full functionality
pip install torch transformers sentence-transformers faiss-cpu scikit-learn

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (HF_TOKEN, OPENAI_API_KEY, etc.)
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up

# Access the application
# - API: http://localhost:8000
# - Gradio Demo: http://localhost:7860
```

## Usage

### Building the Vector Index

```bash
# Step 1: Process raw CSV data
python -m src.cli process_data \
    --csv-path data/amazon_com_ecommerce.csv \
    --output-dir models/

# Step 2: Generate embeddings and build FAISS index
python -m src.cli build-index \
    --csv-path models/processed_data.csv \
    --output-dir models/ \
    --batch-size 32
```

### Running the Demo

```bash
# Launch Gradio web interface
python app.py

# Access at http://localhost:7860
```

### Querying via CLI

```bash
# Search for products
python -m src.cli search \
    --query "wireless headphones under $100" \
    --k 5 \
    --rerank

# Run with LLM response generation
python -m src.cli search \
    --query "educational toys for kids" \
    --k 5 \
    --llm
```

### Using the API

```bash
# Start the FastAPI server
python -m src.cli serve --host 0.0.0.0 --port 8000

# Query the API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text_query": "wireless headphones",
    "k": 5,
    "generate_response": true
  }'
```

## Deployment Options

### 1. HuggingFace Spaces (Recommended for Demos)

The application is designed for easy deployment to HuggingFace Spaces:

**Lightweight Mode** (Free Tier):
- Uses minimal dependencies (gradio, huggingface_hub, pandas)
- Mock product database for instant startup
- Template-based AI responses (no API costs)
- Ideal for demonstrations and testing

**Full Mode** (GPU Spaces):
- Complete CLIP embeddings and FAISS search
- Real LLM integration via APIs
- Production-ready performance

**Deployment Steps:**
1. Create a new Space at https://huggingface.co/new-space
2. Select Gradio SDK and CPU Basic (free) or GPU
3. Upload `app.py` and `requirements.txt`
4. Add secrets in Space settings (HF_TOKEN, OPENAI_API_KEY)
5. Application auto-deploys

### 2. Docker Deployment

```bash
# Production deployment with Docker
docker build -t ecommerce-rag .
docker run -p 8000:8000 -p 7860:7860 ecommerce-rag
```

### 3. Cloud Platforms

The application can be deployed to:
- AWS ECS/EKS
- Google Cloud Run
- Azure Container Instances
- Heroku
- Railway

Use the included Dockerfile and docker-compose.yml for container-based deployments.

## CI/CD Pipeline

The project includes comprehensive GitHub Actions workflows:

### Continuous Integration (ci.yml)

Runs on every push and pull request:
- Multi-version Python testing (3.9, 3.10, 3.11)
- Code linting (flake8, black, isort)
- Type checking (mypy)
- Security scanning (bandit, safety)
- Test coverage reporting
- Docker image builds
- Automated deployment to staging/production

### Model Training (model-training.yml)

Manual workflow for fine-tuning models:
- CLIP fine-tuning on e-commerce data
- Embedding fusion training
- Weights & Biases experiment tracking
- Automatic model upload to HuggingFace Hub

**Trigger Example:**
```bash
gh workflow run model-training.yml \
  -f model_type=clip_finetuning \
  -f data_path=data/amazon_com_ecommerce.csv \
  -f epochs=5
```

### Model Sync (model-sync.yml)

Uploads trained models to HuggingFace Hub:
- Manual or scheduled execution
- Model validation before upload
- Automatic model card generation
- Registry updates

### Release Pipeline (release.yml)

Triggered on version tags (v*):
- Automated PyPI package publishing
- Docker image releases with version tags
- GitHub release creation
- Documentation updates

**Create Release:**
```bash
git tag v1.0.0
git push origin v1.0.0
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Model Configuration
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
EMBEDDING_DIMENSION=512

# LLM Configuration
LLM_PROVIDER=huggingface  # or openai, ollama
LLM_MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1
LLM_API_TOKEN=your_huggingface_token

# Optional: OpenAI
OPENAI_API_KEY=your_openai_key

# FAISS Configuration
FAISS_INDEX_TYPE=flat  # or ivf, hnsw
FAISS_METRIC=l2        # or ip (inner product)

# Paths
FAISS_INDEX_PATH=models/product_index.faiss
FAISS_METADATA_PATH=models/product_metadata.pkl
```

### Configuration File

Alternatively, use `configs/default.yaml`:

```yaml
embedding:
  model_name: "openai/clip-vit-base-patch32"
  dimension: 512

vector_db:
  index_type: "flat"
  metric: "l2"
  nlist: 100
  nprobe: 10

llm:
  provider: "huggingface"
  model_name: "mistralai/Mixtral-8x7B-Instruct-v0.1"
  max_tokens: 300
  temperature: 0.1
```

## Performance

**Benchmarks on Apple M1 with 10K products:**

| Metric | Value |
|--------|-------|
| Search Latency | < 100ms |
| Throughput | 15+ queries/second |
| Index Size | ~20MB for 10K products |
| LLM Response Time | < 2s (API), < 5s (local) |
| Startup Time | ~5s (local), ~30s (Spaces) |
| Memory Usage | ~2GB (full), ~200MB (demo) |

**FAISS Index Types:**

- **Flat**: Exact search, best for < 100K vectors
- **IVF**: Approximate search with clustering, medium-large datasets
- **HNSW**: Fast approximate search, large-scale deployments

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rag_pipeline.py

# Run with verbose output
pytest -v
```

## Data

The project includes Amazon e-commerce product data:

- **amazon_com_ecommerce.csv**: 10,000 products (19MB)
- **test_sample_1000.csv**: 1,000 products (1.9MB) for testing

**Dataset Features:**
- Product names, descriptions, specifications
- Category hierarchies
- Pricing information
- Image URLs
- Amazon product URLs

See [data/README.md](data/README.md) for detailed dataset documentation.

## API Reference

### REST Endpoints

**POST /query** - Search products and generate recommendations

```json
{
  "text_query": "wireless headphones under $100",
  "k": 5,
  "rerank": true,
  "generate_response": true
}
```

**GET /health** - Health check endpoint

**GET /stats** - Index statistics

**Full API documentation** available at `/docs` when server is running.

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting:
   ```bash
   pytest
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

This project follows:
- PEP 8 style guide
- Black code formatting
- isort import sorting
- Type hints for all functions

## Troubleshooting

### Common Issues

**FAISS not installing:**
```bash
# Use CPU version
pip install faiss-cpu

# Or GPU version if CUDA available
pip install faiss-gpu
```

**Model download fails:**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
```

**Out of memory:**
```bash
# Reduce batch size
python -m src.cli build-index --batch-size 16

# Or use smaller sample
python -m src.cli build-index --csv-path data/test_sample_1000.csv
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using the HuggingFace ecosystem
- CLIP model by OpenAI
- FAISS by Facebook Research
- Gradio for web interfaces

## Links

- **GitHub Repository**: https://github.com/ycnliu/ecommerce-rag-pipeline
- **Documentation**: [Full documentation](https://github.com/ycnliu/ecommerce-rag-pipeline/blob/main/README.md)
- **Issues**: [Report bugs or request features](https://github.com/ycnliu/ecommerce-rag-pipeline/issues)

---

This project demonstrates a production-ready RAG pipeline architecture for e-commerce applications, combining state-of-the-art embedding models, efficient vector search, and modern LLMs for intelligent product recommendations.
