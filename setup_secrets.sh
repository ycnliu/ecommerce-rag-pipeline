#!/bin/bash

echo "🔐 GitHub Secrets Setup Helper"
echo "==============================="
echo ""
echo "This script will help you set up GitHub secrets for your CI/CD pipeline."
echo "You'll need to have the tokens/keys ready before running this."
echo ""

# Check if gh CLI is authenticated
if ! gh auth status >/dev/null 2>&1; then
    echo "❌ Error: GitHub CLI not authenticated"
    echo "Please run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI authenticated"
echo ""

# Function to set secret
set_secret() {
    local secret_name=$1
    local description=$2
    local required=$3

    echo "🔑 Setting up: $secret_name"
    echo "Description: $description"

    if [ "$required" = "true" ]; then
        echo "⚠️  This secret is REQUIRED for full functionality"
    else
        echo "ℹ️  This secret is optional"
    fi

    read -p "Do you want to set this secret now? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Enter the secret value (input will be hidden):"
        read -s secret_value

        if [ -n "$secret_value" ]; then
            if gh secret set "$secret_name" --body "$secret_value" >/dev/null 2>&1; then
                echo "✅ Secret $secret_name set successfully"
            else
                echo "❌ Failed to set secret $secret_name"
            fi
        else
            echo "⚠️  Empty value, skipping $secret_name"
        fi
    else
        echo "⏭️  Skipping $secret_name"
    fi
    echo ""
}

echo "📋 Available secrets to configure:"
echo ""

# Required secrets
echo "🔴 REQUIRED SECRETS:"
set_secret "HF_TOKEN" "Hugging Face API token for model hub integration" true

# Optional secrets
echo "🟡 OPTIONAL SECRETS:"
set_secret "WANDB_API_KEY" "Weights & Biases API key for experiment tracking" false
set_secret "DOCKER_USERNAME" "Docker Hub username for container registry" false
set_secret "DOCKER_PASSWORD" "Docker Hub password/token" false
set_secret "PYPI_API_TOKEN" "PyPI API token for package publishing" false
set_secret "TEST_PYPI_API_TOKEN" "Test PyPI API token" false
set_secret "SLACK_WEBHOOK_URL" "Slack webhook URL for notifications" false

echo "🎉 Secret setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Check your secrets: https://github.com/ycnliu/ecommerce-rag-pipeline/settings/secrets/actions"
echo "2. Test a workflow: https://github.com/ycnliu/ecommerce-rag-pipeline/actions"
echo "3. Read the full setup guide: SETUP_GUIDE.md"
echo ""
echo "🚀 Your e-commerce RAG pipeline is ready for action!"