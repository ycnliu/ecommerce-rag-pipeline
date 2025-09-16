#!/bin/bash

echo "🚀 GitHub Repository Setup Script"
echo "================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "❌ Error: Please run this script from the ecommerce_rag_pipeline directory"
    exit 1
fi

echo "📋 Current repository status:"
git status --short
echo ""

# Check if GitHub CLI is authenticated
if ! gh auth status >/dev/null 2>&1; then
    echo "🔑 GitHub Authentication Required"
    echo "Please authenticate with GitHub using one of these methods:"
    echo ""
    echo "Option 1 - Web Authentication:"
    echo "  gh auth login --web"
    echo ""
    echo "Option 2 - Token Authentication:"
    echo "  gh auth login --with-token"
    echo "  (Then paste your GitHub Personal Access Token)"
    echo ""
    echo "Option 3 - Manual Setup:"
    echo "  1. Create a new repository on GitHub.com: https://github.com/new"
    echo "  2. Repository name: ecommerce-rag-pipeline"
    echo "  3. Make it public"
    echo "  4. Don't initialize with README (we have one)"
    echo "  5. Copy the repository URL and run:"
    echo "     git remote add origin https://github.com/YOUR_USERNAME/ecommerce-rag-pipeline.git"
    echo "     git push -u origin main"
    echo ""
    read -p "Press Enter after you've authenticated or set up manually..."
fi

# Try to create repository with GitHub CLI
if gh auth status >/dev/null 2>&1; then
    echo "✅ GitHub CLI authenticated successfully!"
    echo ""
    echo "🏗️ Creating GitHub repository..."

    if gh repo create ecommerce-rag-pipeline --public --source=. --remote=origin --push; then
        echo ""
        echo "🎉 Repository created and pushed successfully!"
        echo "📋 Repository URL: $(gh repo view --web --json url -q .url)"
        echo ""
        echo "🔧 Next steps:"
        echo "1. Set up GitHub Secrets for CI/CD:"
        echo "   - Go to: Settings > Secrets and variables > Actions"
        echo "   - Add the following secrets:"
        echo "     * HF_TOKEN (Hugging Face token)"
        echo "     * WANDB_API_KEY (Weights & Biases - optional)"
        echo "     * DOCKER_USERNAME (Docker Hub)"
        echo "     * DOCKER_PASSWORD (Docker Hub)"
        echo "     * PYPI_API_TOKEN (PyPI - optional)"
        echo "     * SLACK_WEBHOOK_URL (Slack notifications - optional)"
        echo ""
        echo "2. Enable GitHub Actions if not already enabled"
        echo "3. Set up branch protection rules (optional)"
        echo "4. Add collaborators (optional)"
    else
        echo "❌ Failed to create repository with GitHub CLI"
        echo "Please create the repository manually:"
        echo "1. Go to: https://github.com/new"
        echo "2. Repository name: ecommerce-rag-pipeline"
        echo "3. Make it public"
        echo "4. Don't initialize with README"
        echo "5. After creation, run:"
        echo "   git remote add origin https://github.com/YOUR_USERNAME/ecommerce-rag-pipeline.git"
        echo "   git push -u origin main"
    fi
else
    echo "❌ GitHub CLI authentication failed"
    echo "Please follow the manual setup instructions above"
fi