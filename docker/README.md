# Docker

## Overview

The docker directory centralizes containerization assets and operational standards for building, testing, and deploying the Real-Time Market Intelligence Platform using Docker and Docker Compose. It provides canonical Dockerfiles, multi-stage builds, environment templates, and CI/CD integration patterns.

## Purpose

- Standardize Docker images across services
- Enable reproducible local and CI builds
- Provide secure defaults and minimal images
- Document deployment patterns to multiple environments
- Offer templates for common operations and orchestration

## Directory Structure

```
docker/
├── base/                   # Base images and shared layers
│   ├── python.Dockerfile  # Python base with system deps
│   └── node.Dockerfile    # Node base for frontend builds
├── services/               # Service-specific Dockerfiles
│   ├── api.Dockerfile     # FastAPI service
│   ├── stream.Dockerfile  # Streaming/worker service
│   └── frontend.Dockerfile# React app (nginx or node serve)
├── compose/                # Compose profiles by environment
│   ├── docker-compose.dev.yml
│   ├── docker-compose.test.yml
│   └── docker-compose.prod.yml
├── ci/                     # CI helpers for builds and scans
│   ├── build.sh           # Build and tag images
│   ├── push.sh            # Push images to registry
│   └── scan.sh            # Security scans (Trivy/Grype)
├── k8s/                    # Optional k8s manifests (if needed)
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
├── templates/              # Reusable templates
│   ├── .dockerignore      # Recommended dockerignore
│   ├── Dockerfile.py      # Template for Python services
│   ├── Dockerfile.node    # Template for Node/Frontend
│   └── entrypoint.sh      # Safe shell entrypoint template
└── README.md               # This file
```

## Image Standards

- Use minimal, verified base images (e.g., python:3.11-slim, node:20-alpine)
- Multi-stage builds to keep runtime images small
- Non-root user by default (e.g., UID 1001)
- Immutable container filesystem where possible
- Pin critical system packages and python/node dependencies
- Expose only required ports
- Healthcheck for critical services
- Labels for SBOM, vcs-ref, build-date, version, maintainer

Example labels:
```
LABEL org.opencontainers.image.source="https://github.com/galafis/real-time-market-intelligence" \
      org.opencontainers.image.revision="$GIT_SHA" \
      org.opencontainers.image.created="$BUILD_DATE"
```

## Python Service Dockerfile Template

```
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Builder for dependencies (wheels)
FROM base AS builder
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# Final runtime
FROM python:3.11-slim AS runtime
RUN useradd -r -u 1001 -g root appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt
COPY src ./src
USER 1001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "src.api.main"]
```

## Frontend Dockerfile Template

```
# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend .
RUN npm run build

# Run stage (nginx)
FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/build /usr/share/nginx/html
COPY docker/templates/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK CMD wget -qO- http://localhost/ || exit 1
```

## docker-compose Profiles

- dev: hot-reload, bind mounts, verbose logs
- test: ephemeral services for CI (services mocked as needed)
- prod: production-like, read-only fs, resource limits

Example overrides:
```
services:
  api:
    build:
      context: .
      dockerfile: docker/services/api.Dockerfile
    env_file: .env
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
```

## CI/CD Patterns

### GitHub Actions: Build and Push Images

```
name: Docker Images
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      - name: Set up Buildx
        uses: docker/setup-buildx-action@v3
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push API image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/services/api.Dockerfile
          push: ${{ github.ref == 'refs/heads/master' }}
          tags: ghcr.io/${{ github.repository }}/api:latest
          cache-from: type=registry,ref=ghcr.io/${{ github.repository }}/api:cache
          cache-to: type=inline
      - name: Trivy scan
        uses: aquasecurity/trivy-action@0.20.0
        with:
          image-ref: ghcr.io/${{ github.repository }}/api:latest
          format: table
          ignore-unfixed: true
```

### Versioning and Tags

- latest for master branch
- semver tags (e.g., v1.2.3)
- git sha annotation label org.opencontainers.image.revision

## Deployment Guidelines

- Use read-only root filesystem and drop Linux capabilities
- Configure resource requests/limits
- Externalize configuration via environment variables and secrets
- Use a private registry (GHCR/ECR/GCR) with least privilege
- Sign images (cosign) and verify in deployment pipeline

## Security Best Practices

- Regularly scan images (Trivy/Grype) and base images
- Avoid embedding secrets in images
- Keep SBOMs (syft) and track vulnerabilities
- Run as non-root and restrict file system writes
- Validate inputs at the API gateway and service layers

## Local Developer Workflow

```
# Build API image
DOCKER_BUILDKIT=1 docker build -f docker/services/api.Dockerfile -t rtm-api:dev .

# Start dev stack
docker compose -f docker/compose/docker-compose.dev.yml up --build

# Run tests inside container
docker compose -f docker/compose/docker-compose.test.yml run --rm api pytest -q
```

## Templates

- templates/Dockerfile.py: Multi-stage, non-root, healthcheck, labels
- templates/Dockerfile.node: Build + runtime separation
- templates/.dockerignore: Ignore venv, node_modules, caches, tests artifacts
- templates/entrypoint.sh: Safe shell with exec, pipefail, and trap on signals

## Registry Conventions

- ghcr.io/OWNER/REPO/SERVICE:TAG
- Include platform builds if needed (linux/amd64, linux/arm64)
- Use cache-from to accelerate CI builds

## Troubleshooting

- Build context too large: verify .dockerignore
- Permission denied: ensure USER matches volume ownership or use fsGroup
- Long cold starts: enable Buildx cache and layer reuse
- Port conflicts: adjust host ports in compose profiles

## Next Steps

- Add service Dockerfiles under docker/services/
- Provide compose files for dev/test/prod
- Integrate CI pipeline for image build, scan, and push
- Document Kubernetes manifests if applicable
