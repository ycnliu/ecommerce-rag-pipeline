#!/bin/bash

# E-commerce RAG Pipeline Setup Script

set -e  # Exit on any error

echo "🚀 Setting up E-commerce RAG Pipeline..."

# Check if Python 3.9+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.9+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "📦 Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p models/cache
mkdir -p logs
mkdir -p configs

# Copy example configuration
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment configuration..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
else
    echo "⚙️  Environment configuration already exists"
fi

# Check if NLTK data is available for evaluation
echo "📚 Downloading NLTK data for evaluation..."
python3 -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print('NLTK punkt tokenizer already available')
except LookupError:
    print('Downloading NLTK punkt tokenizer...')
    nltk.download('punkt', quiet=True)
" || echo "⚠️  NLTK download failed, evaluation features may not work"

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🔧 Next steps:"
echo "1. Edit .env file with your configuration (LLM API token, data paths, etc.)"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Check status: python -m src.cli status"
echo "4. Process your data: python -m src.cli process-data -i /path/to/your/data.csv"
echo "5. Build search index: python -m src.cli build-index -i data/processed_data.csv"
echo "6. Start the API server: python -m src.cli serve"
echo ""
echo "📖 For more commands: python -m src.cli --help"