---
title: E-commerce RAG Pipeline
emoji: 🛒
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# E-commerce RAG Pipeline - Demo

This is an automated deployment of the E-commerce RAG Pipeline demo.

## Deployment Info

- **Deployed via:** GitHub Actions CI/CD pipeline
- **Quality Gates:** All tests, linting, and security scans passed
- **Source Repository:** [github.com/ycnliu/ecommerce-rag-pipeline](https://github.com/ycnliu/ecommerce-rag-pipeline)

## About

This demo showcases a production-ready RAG (Retrieval-Augmented Generation) pipeline for e-commerce product search and recommendations.

### Features (Demo Version)

- Product search interface
- Template-based AI recommendations
- Lightweight deployment (CPU-only)

### Full Version Features

The complete system (available in the GitHub repository) includes:

- **CLIP Embeddings**: Multimodal text-image embeddings (512-dim)
- **FAISS Vector Search**: Efficient similarity search across 10K products
- **LLM Integration**: Real AI responses via OpenAI/HuggingFace APIs
- **Advanced RAG**: Context retrieval and prompt engineering
- **CI/CD Pipeline**: Automated testing, quality gates, and deployment

## Architecture

```
User Query → Embedding Generation → Vector Search → Context Retrieval → LLM Generation → Response
```

## Deployment Pipeline

This Space is automatically deployed through:

1. **Quality Gates**: Tests, linting, type checking, security scans
2. **Build Artifacts**: Immutable demo dataset and metadata
3. **Preview Deployment**: PR-based preview environments
4. **Production Deployment**: Main branch deployments with smoke tests

## Links

- **Full Documentation**: [README.md](https://github.com/ycnliu/ecommerce-rag-pipeline/blob/main/README.md)
- **Source Code**: [GitHub Repository](https://github.com/ycnliu/ecommerce-rag-pipeline)
- **CI/CD Workflows**: [.github/workflows/](https://github.com/ycnliu/ecommerce-rag-pipeline/tree/main/.github/workflows)

---

*This demo is part of a production-ready RAG pipeline with enterprise CI/CD practices.*
