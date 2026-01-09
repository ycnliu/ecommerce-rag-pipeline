# CI/CD Architecture

This document outlines the enterprise-grade CI/CD pipeline implemented for the E-commerce RAG Pipeline project.

## Overview

The project implements a production-ready CI/CD strategy that treats HuggingFace Spaces as a proper deployment target with full quality gates, security scanning, and controlled releases.

## Key Principles

1. **Build Once, Deploy Many**: Artifacts are built once and deployed to multiple environments
2. **Quality Gates**: All changes must pass tests, linting, type checking, and security scans
3. **Immutable Deployments**: Deploy from versioned artifacts with checksums
4. **Environment Promotion**: Preview (PR) → Production (main) → Release (tag)
5. **Automated Verification**: Post-deployment smoke tests ensure service health
6. **Supply Chain Security**: SBOM generation, dependency scanning, secret detection

## Pipeline Architecture

```
Code Push/PR
    ↓
┌─────────────────────┐
│   Quality Gates     │
│  - Tests (3.9-3.11) │
│  - Lint (ruff)      │
│  - Type check (mypy)│
│  - Security scans   │
│  - SBOM generation  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Build Artifacts    │
│  - demo_products    │
│  - manifest.json    │
│  - checksums        │
└──────────┬──────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
Preview Env   Production Env
(PR Space)    (Main Space)
    ↓             ↓
Smoke Test    Smoke Test
```

## Workflows

### 1. Continuous Integration (ci.yml)

**Triggers**: Every push to main/develop, all PRs

**Jobs**:

- **Test Matrix**
  - Python 3.9, 3.10, 3.11
  - pytest with coverage (70% minimum)
  - Code formatting (black, isort)
  - Linting (flake8/ruff)
  - Type checking (mypy)

- **Security Scanning**
  - Bandit (code security)
  - pip-audit (dependency vulnerabilities)
  - Gitleaks (secret scanning)
  - SBOM generation (Syft)

- **Docker Build**
  - Multi-architecture (amd64, arm64)
  - Only on main branch

### 2. Deploy to Spaces (deploy-spaces.yml)

**Triggers**: PRs, pushes to main

**Environments**:
- **Preview**: `ycnliu/ecommerce-rag-preview` (PRs only)
- **Production**: `ycnliu/ecommerce-rag-demo` (main branch)

**Jobs**:

1. **Build Artifacts**
   - Extract 100 demo products
   - Generate manifest with file hashes
   - Upload as artifacts (30-day retention)

2. **Quality Gates**
   - All tests pass
   - Linting pass
   - Security scans pass

3. **Deploy Preview** (PRs)
   - Deploy to preview Space
   - Smoke test deployment
   - Comment PR with preview URL

4. **Deploy Production** (main)
   - Deploy to production Space
   - Extended smoke test
   - Verify Space health

**Deployment Gates**: Tests + lint + type check + security must pass

**Post-Deploy Verification**:
- HTTP health check
- Content verification
- Response time check

### 3. Release Pipeline (release.yml)

**Triggers**: Git tags (v*)

**Process**:

1. **Create Release**
   - Generate changelog
   - Create GitHub Release
   - Tag version

2. **Build & Publish**
   - Build Python package
   - Test on Test PyPI
   - Publish to PyPI

3. **Docker Release**
   - Build multi-arch images
   - Tag with version + latest
   - Push to Docker Hub

4. **Deploy to Spaces**
   - Build release artifacts
   - Deploy to production Space
   - Verify deployment

5. **Update Documentation**
   - Generate API docs (Sphinx)
   - Deploy to GitHub Pages

**Release Strategy**: Only from git tags - enforces controlled releases

**Example**:
```bash
git tag v1.0.0
git push origin v1.0.0
# Triggers full release pipeline
```

### 4. Model Training (model-training.yml)

**Triggers**: Manual dispatch

**Process**:
- CLIP fine-tuning on e-commerce data
- Experiment tracking (Weights & Biases)
- Model evaluation
- Upload to HuggingFace Hub

### 5. Model Sync (model-sync.yml)

**Status**: Disabled (scheduled runs commented out)

**Purpose**: Sync trained models to HuggingFace Hub

**Note**: Currently disabled until models are trained. Will auto-sync when models exist.

## Security Features

### Dependency Management

**Dependabot** (.github/dependabot.yml):
- Weekly scans for Python dependencies
- Weekly scans for GitHub Actions
- Auto-creates PRs for updates
- Labels: dependencies, python, github-actions

### Secret Scanning

**Gitleaks**: Scans all commits for leaked secrets
- API keys
- Passwords
- Tokens
- Private keys

### Vulnerability Scanning

- **Bandit**: Static code analysis for security issues
- **pip-audit**: Known vulnerabilities in dependencies
- **Safety**: Comprehensive dependency audit

### Supply Chain Security

**SBOM (Software Bill of Materials)**:
- Generated with Syft
- SPDX format
- Artifact uploaded with every build
- Version-tracked with commit SHA

## Space Deployment Strategy

### Lightweight Demo Approach

The `space/` folder contains a minimal demo version:

```
space/
├── app.py              # Lightweight Gradio app
├── requirements.txt    # Minimal deps (gradio only)
├── README.md          # Space metadata
├── demo_products.csv  # Build artifact (100 products)
└── manifest.json      # Build artifact (checksums)
```

**Why separate from main app**:
- Main repo: Full pipeline with CLIP, FAISS, LLMs
- Space demo: Lightweight, fast startup, CPU-only
- Artifact immutability: Built in CI, deployed unchanged

### Build Once Philosophy

**Artifacts built in CI**:
1. `demo_products.csv` - Processed subset of data
2. `manifest.json` - File hashes and metadata
3. Checksums for verification

**Deployed as-is**: No rebuilding in Space, ensures consistency

### Environment Promotion

```
PR → Preview Space (ephemeral)
    ↓ merge
Main → Production Space (demo)
    ↓ tag
Tag → Production Space (release)
```

## Quality Gates

All deployments must pass:

1. **Tests**: All unit tests pass, coverage > 70%
2. **Lint**: Code formatting and style checks
3. **Type Check**: mypy type checking
4. **Security**: Bandit + pip-audit + gitleaks
5. **Build**: Artifacts build successfully
6. **Deploy**: Smoke tests pass post-deployment

## Monitoring & Verification

### Post-Deploy Checks

**Smoke Tests**:
- HTTP 200 response
- Expected content verification
- Response time < 30s

**Verification Script** (verify_space.py):
- Retries with exponential backoff
- Content validation
- Error reporting

### Artifact Tracking

**Manifest File**:
```json
{
  "version": "abc123",
  "generated_at": "2024-01-09T...",
  "files": {
    "app.py": {
      "hash": "sha256:...",
      "size": 12345,
      "modified": "..."
    }
  }
}
```

## Best Practices Implemented

1. **Immutable Artifacts**: Build once, deploy many times
2. **Ephemeral Environments**: PR-based preview deployments
3. **Quality Gates**: Automated checks before deployment
4. **Post-Deploy Verification**: Smoke tests ensure service health
5. **Security Scanning**: SBOM, secret detection, vulnerability scans
6. **Controlled Releases**: Tag-based releases only
7. **Artifact Traceability**: SHA hashes and manifests
8. **Environment Parity**: Same artifacts across preview/production

## GitHub Secrets Required

| Secret | Purpose | Required |
|--------|---------|----------|
| `HF_TOKEN` | HuggingFace deployment | Yes |
| `OPENAI_API_KEY` | Optional LLM enhancement | No |
| `DOCKER_USERNAME` | Docker Hub publishing | Optional |
| `DOCKER_PASSWORD` | Docker Hub auth | Optional |
| `PYPI_API_TOKEN` | Package publishing | Optional |
| `SLACK_WEBHOOK_URL` | Notifications | Optional |

## Metrics

**What this pipeline demonstrates**:

- Multi-environment deployment (preview/prod/release)
- Artifact immutability with checksums
- Automated quality gates
- Security-first approach (SBOM, secret scanning)
- Post-deployment verification
- Controlled release process
- Supply chain transparency

**Enterprise patterns**:
- GitOps-style deployments
- Environment promotion strategy
- Build artifact traceability
- Automated compliance (SBOM)
- Defense in depth (multiple security layers)

## Future Enhancements

Possible additions:
- Blue/green deployments
- Canary releases with traffic splitting
- Performance regression testing
- Load testing in preview
- Automated rollback on failed smoke tests
- Slack/email notifications
- Deployment dashboards

---

This CI/CD architecture provides enterprise-grade deployment practices while keeping the HuggingFace Spaces demo lightweight and accessible.
