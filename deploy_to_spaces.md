# 🚀 Deploy to HuggingFace Spaces

## Quick Deployment Guide

### Option 1: Web Interface (Recommended)

1. **Go to HuggingFace Spaces**: https://huggingface.co/spaces
2. **Click "Create new Space"**
3. **Fill in details**:
   - **Space name**: `ecommerce-rag-pipeline`
   - **License**: `MIT`
   - **Space SDK**: `Gradio`
   - **Visibility**: `Public`
4. **Clone the space**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/ecommerce-rag-pipeline
   cd ecommerce-rag-pipeline
   ```

5. **Copy files from this repo**:
   ```bash
   # Copy essential files
   cp ../ecommerce-rag-pipeline/app.py .
   cp ../ecommerce-rag-pipeline/requirements-spaces.txt requirements.txt
   cp ../ecommerce-rag-pipeline/README-spaces.md README.md

   # Copy source code (optional, for full functionality)
   mkdir -p src
   cp -r ../ecommerce-rag-pipeline/src/* src/
   ```

6. **Commit and push**:
   ```bash
   git add .
   git commit -m "Initial deployment of e-commerce RAG pipeline"
   git push
   ```

### Option 2: Direct Upload

1. **Create ZIP file** with these files:
   - `app.py`
   - `requirements.txt` (renamed from requirements-spaces.txt)
   - `README.md` (renamed from README-spaces.md)
   - `src/` directory (optional)

2. **Upload via web interface** at https://huggingface.co/spaces

### Option 3: CLI Deployment

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Login to HuggingFace
huggingface-cli login --token YOUR_HF_TOKEN

# Create space
huggingface-cli repo create ecommerce-rag-pipeline --type space --space_sdk gradio

# Clone and deploy
git clone https://huggingface.co/spaces/YOUR_USERNAME/ecommerce-rag-pipeline
cd ecommerce-rag-pipeline

# Copy files and push
cp ../app.py .
cp ../requirements-spaces.txt requirements.txt
cp ../README-spaces.md README.md
mkdir -p src && cp -r ../src/* src/

git add .
git commit -m "Deploy e-commerce RAG pipeline to Spaces"
git push
```

## 🔧 Configuration for Spaces

### 1. Minimal Deployment (Demo Mode)
- Uses fallback responses
- No external dependencies
- Fast startup time
- Works immediately

### 2. Full Deployment (Production Mode)
- Uncomment dependencies in requirements.txt
- Add model files to repository
- Set up secrets for API keys
- Enable GPU for better performance

### 3. Environment Variables (Optional)

Add these in Space settings → Repository secrets:

```bash
HF_TOKEN=your_huggingface_token
LLM_PROVIDER=free
LLM_MODEL_NAME=microsoft/DialoGPT-medium
DEVICE=cpu
```

## 📋 Post-Deployment Checklist

- [ ] Space loads without errors
- [ ] Search functionality works
- [ ] AI responses are generated
- [ ] Interface is responsive
- [ ] Examples work correctly
- [ ] Links point to correct repositories

## 🐛 Troubleshooting

### Common Issues:

1. **Space won't start**:
   - Check requirements.txt for typos
   - Ensure app.py has no syntax errors
   - Remove heavy dependencies for initial deployment

2. **Import errors**:
   - Comment out optional imports in app.py
   - Use fallback mode for demonstration

3. **Slow loading**:
   - Remove large model files
   - Use CPU inference initially
   - Add loading progress indicators

4. **Memory errors**:
   - Reduce batch sizes
   - Use smaller models
   - Request GPU upgrade

## 🚀 Going Live

Once deployed, your space will be available at:
`https://huggingface.co/spaces/YOUR_USERNAME/ecommerce-rag-pipeline`

Share the link and get feedback from users!

## 🔄 Updates

To update your deployed space:

```bash
cd ecommerce-rag-pipeline
# Make changes to files
git add .
git commit -m "Update: description of changes"
git push
```

The space will automatically rebuild and deploy.

## 📊 Analytics

Monitor your space performance:
- View usage stats in HuggingFace dashboard
- Check logs for errors
- Monitor response times
- Gather user feedback

## 🎯 Next Steps

1. **Deploy the demo** → Get immediate feedback
2. **Add GPU support** → Improve response quality
3. **Enable model uploads** → Use custom fine-tuned models
4. **Add authentication** → Control access
5. **Integrate analytics** → Track usage patterns