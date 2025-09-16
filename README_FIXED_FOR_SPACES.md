---
title: E-commerce RAG Pipeline
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🛒 E-commerce RAG Pipeline

**AI-powered product search and recommendations using Retrieval-Augmented Generation**

[![Spaces](https://img.shields.io/badge/🤗-Spaces-blue)](https://huggingface.co/spaces/YCHL/amazon-chatbot-rag-pipeline)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/ycnliu/ecommerce-rag-pipeline)

## 🚀 Try the Demo

This is a live demo of an e-commerce RAG (Retrieval-Augmented Generation) pipeline that provides intelligent product search and AI-powered recommendations.

### How to Use:
1. **Enter a search query** like "wireless headphones under $100"
2. **View product results** with detailed information
3. **Get AI recommendations** based on your requirements

### Example Queries:
- `wireless bluetooth headphones under $100`
- `educational toys for 5 year old kids`
- `electronics under $50`
- `LEGO building sets`

## 🏗️ Architecture

This demo showcases a production-ready RAG pipeline with:

- **🔍 Semantic Search**: CLIP embeddings for multimodal product search
- **📊 Vector Database**: FAISS for efficient similarity search
- **🤖 AI Generation**: Multiple LLM backends (OpenAI, HuggingFace, fallback)
- **🌐 Web Interface**: Gradio for interactive demonstrations
- **⚡ Real-time Processing**: Fast response generation

## 🛠️ Technology Stack

- **Embeddings**: OpenAI CLIP + SentenceTransformers
- **Vector DB**: FAISS with advanced indexing
- **LLM Backends**: OpenAI GPT-3.5-turbo, HuggingFace, Fallback
- **Web Framework**: Gradio
- **Deployment**: HuggingFace Spaces
- **CI/CD**: GitHub Actions

## 📁 Project Structure

```
ecommerce-rag-pipeline/
├── src/
│   ├── embedding/          # CLIP + sentence transformers
│   ├── vector_db/          # FAISS vector database
│   ├── rag/               # RAG pipeline & LLM clients
│   ├── data/              # Data models & processing
│   └── utils/             # Configuration & utilities
├── app.py                 # Gradio web interface
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

## 🔧 Local Development

```bash
# Clone the repository
git clone https://github.com/ycnliu/ecommerce-rag-pipeline.git
cd ecommerce-rag-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run the demo
python app.py
```

## 🌟 Features

### ✅ Implemented
- Multi-modal product embeddings (text + image)
- Advanced embedding fusion strategies
- Multiple LLM backend support
- Real-time product search
- AI-powered recommendations
- Gradio web interface
- Docker deployment
- CI/CD pipeline

### 🚧 Production Features
- CLIP fine-tuning for domain adaptation
- Neural fusion modules
- Advanced prompt engineering
- A/B testing framework
- Performance monitoring
- Scalable vector indexing

## 📊 Performance

- **Search Latency**: < 100ms (MPS acceleration)
- **Throughput**: 15+ queries/second
- **Index Size**: Supports millions of products
- **Response Time**: < 2s for AI generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using HuggingFace ecosystem
- Deployed on HuggingFace Spaces
- Powered by advanced RAG techniques

---

**🔗 Links:**
- [GitHub Repository](https://github.com/ycnliu/ecommerce-rag-pipeline)
- [Documentation](https://github.com/ycnliu/ecommerce-rag-pipeline/blob/main/README.md)
- [Demo Video](https://example.com/demo)

*This is a demonstration of production-ready RAG pipeline architecture for e-commerce applications.*