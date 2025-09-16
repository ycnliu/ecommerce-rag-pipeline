# 🚀 Complete Setup Guide

Welcome to your E-commerce RAG Pipeline! Your code has been successfully pushed to GitHub. Follow these steps to complete the setup.

## 📍 Repository Information
- **GitHub Repository:** https://github.com/ycnliu/ecommerce-rag-pipeline
- **Status:** ✅ Code pushed successfully
- **Branch:** main
- **Files:** 47 files committed

## 🔧 Required GitHub Secrets Setup

To enable the full CI/CD pipeline, you need to add these secrets to your GitHub repository:

### 1. Navigate to Secrets Settings
1. Go to your repository: https://github.com/ycnliu/ecommerce-rag-pipeline
2. Click on **Settings** tab
3. Go to **Secrets and variables** > **Actions**
4. Click **New repository secret** for each secret below

### 2. Required Secrets

#### 🤗 Hugging Face Integration (REQUIRED)
```
Name: HF_TOKEN
Value: [Your Hugging Face API Token]
```
- Get your token: https://huggingface.co/settings/tokens
- Needed for: Model uploads, downloads, and hub integration

#### 📊 Weights & Biases (Optional - for experiment tracking)
```
Name: WANDB_API_KEY
Value: [Your W&B API Key]
```
- Get your key: https://wandb.ai/authorize
- Needed for: Training experiment logging and monitoring

#### 🐳 Docker Hub (Optional - for container registry)
```
Name: DOCKER_USERNAME
Value: [Your Docker Hub username]

Name: DOCKER_PASSWORD
Value: [Your Docker Hub password/token]
```
- Sign up: https://hub.docker.com/
- Needed for: Automated Docker image builds and publishing

#### 📦 PyPI (Optional - for package publishing)
```
Name: PYPI_API_TOKEN
Value: [Your PyPI API Token]

Name: TEST_PYPI_API_TOKEN
Value: [Your Test PyPI API Token]
```
- Get tokens: https://pypi.org/manage/account/
- Needed for: Automated package releases

#### 💬 Slack Notifications (Optional)
```
Name: SLACK_WEBHOOK_URL
Value: [Your Slack Webhook URL]
```
- Set up: https://api.slack.com/messaging/webhooks
- Needed for: CI/CD notifications

## 🔄 GitHub Actions Workflows

Your repository includes 4 automated workflows:

### 1. **Continuous Integration** (`.github/workflows/ci.yml`)
- **Triggers:** Push to main/develop, Pull Requests
- **Features:**
  - Multi-Python testing (3.9, 3.10, 3.11)
  - Code quality checks (flake8, black, isort, mypy)
  - Security scanning (bandit, safety)
  - Docker builds
  - Coverage reporting

### 2. **Model Training** (`.github/workflows/model-training.yml`)
- **Triggers:** Manual dispatch
- **Features:**
  - Automated CLIP fine-tuning
  - Embedding fusion training
  - W&B experiment tracking
  - Model evaluation and storage

### 3. **Model Sync** (`.github/workflows/model-sync.yml`)
- **Triggers:** Manual dispatch, Daily schedule (2 AM UTC)
- **Features:**
  - Automatic model uploads to Hugging Face Hub
  - Model validation and registry updates
  - Notification system

### 4. **Release Pipeline** (`.github/workflows/release.yml`)
- **Triggers:** Git tags (v*)
- **Features:**
  - PyPI package publishing
  - Docker image releases
  - Documentation updates
  - GitHub releases

## 🚀 Quick Start Commands

### Local Development
```bash
# Clone your repository
git clone https://github.com/ycnliu/ecommerce-rag-pipeline.git
cd ecommerce-rag-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API tokens

# Process data and build index
python -m src.cli process-data -i data/your_products.csv -o models/
python -m src.cli build-index -i models/processed_data.csv -o models/

# Start API server
python -m src.cli serve --host 0.0.0.0 --port 8000
```

### Trigger Model Training
```bash
# Using GitHub CLI
gh workflow run model-training.yml \
  -f model_type=clip_finetuning \
  -f data_path=data/amazon_com_ecommerce.csv \
  -f epochs=5 \
  -f use_wandb=true

# Using GitHub Web Interface
# 1. Go to Actions tab
# 2. Select "Model Training Pipeline"
# 3. Click "Run workflow"
# 4. Fill in parameters
```

### Create a Release
```bash
# Tag and push for automatic release
git tag v1.0.0
git push origin v1.0.0

# Or use GitHub CLI
gh release create v1.0.0 --title "Initial Release" --notes "First stable release"
```

## 🔍 Monitoring and Logs

### GitHub Actions
- **View runs:** https://github.com/ycnliu/ecommerce-rag-pipeline/actions
- **Check logs:** Click on any workflow run
- **Enable/disable:** Actions tab > Workflow > Enable/Disable

### Model Training
- **W&B Dashboard:** https://wandb.ai/your-username/ecommerce-rag-pipeline
- **Model Registry:** Check `model_registry.json` in your repo
- **Hugging Face Models:** https://huggingface.co/your-username

### API Monitoring
- **Health Check:** `GET /health`
- **Metrics:** `GET /stats`
- **Documentation:** http://localhost:8000/docs (when running locally)

## 📚 Documentation

### API Documentation
- **Interactive Docs:** Available at `/docs` when server is running
- **OpenAPI Spec:** Available at `/openapi.json`

### Code Documentation
- **README:** Comprehensive usage guide
- **Docstrings:** All functions and classes documented
- **Type Hints:** Full type annotations

## 🛠️ Development Workflow

### Making Changes
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
python -m pytest tests/
python -m src.cli status

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# Create pull request on GitHub
gh pr create --title "Add new feature" --body "Description"
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Security check
bandit -r src/
```

## 🔧 Troubleshooting

### Common Issues
1. **GitHub Actions failing:** Check secrets are properly set
2. **Model training timeout:** Increase timeout in workflow or use smaller datasets
3. **Docker build fails:** Ensure Docker secrets are correct
4. **API errors:** Check environment variables and dependencies

### Getting Help
- **Issues:** https://github.com/ycnliu/ecommerce-rag-pipeline/issues
- **Discussions:** https://github.com/ycnliu/ecommerce-rag-pipeline/discussions
- **Documentation:** Check README.md and code comments

## 🎉 What's Next?

1. ✅ **Set up GitHub secrets** (follow section 2 above)
2. 🧪 **Test the workflows** by triggering a model training run
3. 📊 **Set up monitoring** with W&B and/or Slack
4. 🚀 **Deploy to production** using Docker or cloud platforms
5. 📈 **Monitor performance** and iterate on your models

Your e-commerce RAG pipeline is now ready for production use! 🚀