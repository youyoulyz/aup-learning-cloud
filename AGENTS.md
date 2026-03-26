# AGENTS.md - AUP Learning Cloud

Guidelines for AI agents working on this codebase.

## Project Overview

AUP Learning Cloud is a JupyterHub deployment for AI education on AMD hardware (GPU/NPU/CPU). Stack: Python, TypeScript/React, Docker, Kubernetes (K3s), Helm, Ansible.

## Build/Lint/Test Commands

### All-in-One
```bash
# Run all checks (pre-commit)
pre-commit run --all-files

# Install pre-commit hooks
pip install pre-commit && pre-commit install
```

### Python (Ruff)
```bash
# Lint check
ruff check .

# Auto-fix issues
ruff check --fix .

# Format check
ruff format --check .

# Format
ruff format .
```

### Frontend (from runtime/hub/frontend/)
```bash
cd runtime/hub/frontend

# Install deps
pnpm install

# Lint (ESLint)
pnpm run lint

# Type check (TypeScript)
pnpm run build

# Format check
pnpm run format:check

# Format
pnpm run format
```

### Shell Scripts
```bash
# Lint with ShellCheck
find . -name "*.sh" -o -name "*.bash" | \
  grep -v node_modules | grep -v .git | \
  xargs -r shellcheck
```

### YAML
```bash
yamllint .
```

## Code Style Guidelines

### Python
- **Target**: Python 3.10+ (`pyproject.toml`)
- **Line length**: 120 characters
- **Indentation**: 4 spaces
- **Quotes**: Double quotes preferred
- **Imports**: Sorted automatically (isort), stdlib → third-party → first-party
- **Docstrings**: Google-style or reStructuredText, all public functions
- **Types**: Use type hints for function signatures
- **Naming**: snake_case (functions/vars), PascalCase (classes), UPPER_CASE (consts)
- **Line endings**: LF (Unix-style)

**Notebook Exceptions** (`projects/**/*.ipynb`):
- Imports can be at any cell (not just top)
- `from module import *` allowed for teaching
- Single-letter variables (x, y, l) permitted
- Unused variables allowed (exploratory code)

### TypeScript/React (Frontend)
- **Indentation**: 2 spaces
- **Line length**: 120 characters
- **Framework**: React 18+ with hooks
- **Components**: PascalCase, function components preferred
- **Imports**: Absolute imports for shared packages (`@auplc/shared`)
- **Line endings**: LF (Unix-style)
- **File naming**: kebab-case for files, PascalCase for components

### Shell Scripts
- **Dialect**: Bash (`.shellcheckrc`)
- **Indentation**: 2 spaces (see `.editorconfig`)
- **Shebang**: `#!/usr/bin/env bash`
- **Safety**: Use `set -euo pipefail` for new scripts
- **Quoting**: Always quote variables (`"$var"`)

### YAML
- **Indentation**: 2 spaces
- **Line length**: 200 max (relaxed for K8s manifests)
- **Comments**: Space after `#`
- **Truthy**: Allows `yes/no/on/off` (Ansible compatibility)

## Project Structure

```
├── projects/          # Educational notebooks (CV, DL, LLM, PhySim)
├── runtime/           # JupyterHub deployment
│   ├── hub/          # Hub config, frontend apps, core Python
│   └── chart/        # Helm chart
├── deploy/           # Ansible playbooks for cluster setup
├── dockerfiles/      # Container image definitions
├── scripts/          # Utility scripts (Python/bash)
└── auplc-installer   # Main deployment script
```

## Git Workflow

### Branch Naming
| Prefix | Use Case |
|--------|----------|
| `feature/` | New features |
| `bugfix/` | Bug fixes |
| `hotfix/` | Critical production fixes |
| `refactor/` | Code restructuring |
| `docs/` | Documentation |
| `chore/` | Build/config changes |

### Commit Messages
- Use imperative mood ("Add feature", not "Added feature")
- First line under 72 characters
- Reference issues when applicable

## Important Notes

1. **VSCode Settings**: Use `.vscode/settings.json` for consistent formatting
2. **No tests currently**: This project doesn't use pytest/jest
3. **Docker builds**: Use `dockerfiles/Makefile` with `GPU_TARGET` env var
4. **Jupyter notebooks**: Relaxed rules in `projects/`, strict rules elsewhere
5. **Trailing newlines**: All files must end with newline (enforced)

## Configuration Files

- **Python**: `pyproject.toml` (Ruff settings)
- **YAML**: `.yamllint.yaml`
- **Shell**: `.shellcheckrc`
- **Editor**: `.editorconfig`
- **Git hooks**: `.pre-commit-config.yaml`
