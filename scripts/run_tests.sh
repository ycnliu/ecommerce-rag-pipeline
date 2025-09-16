#!/bin/bash

# E-commerce RAG Pipeline Test Runner

set -e

echo "🧪 Running E-commerce RAG Pipeline Tests..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing test dependencies..."
    pip install pytest pytest-asyncio httpx
fi

# Run tests with coverage if available
if command -v pytest-cov &> /dev/null; then
    echo "📊 Running tests with coverage..."
    pytest tests/ \
        --cov=src \
        --cov-report=html \
        --cov-report=term-missing \
        --cov-fail-under=70 \
        -v

    echo "📈 Coverage report generated in htmlcov/"
else
    echo "🧪 Running tests..."
    pytest tests/ -v
fi

echo "✅ Tests completed successfully!"