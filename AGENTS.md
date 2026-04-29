<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved. -->
<!--
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# AGENTS.md — AUP Learning Cloud

This file provides guidance for AI coding agents (Cursor, Claude Code, GitHub Copilot Workspace,
Devin, and similar tools) working in this repository.

## Project Identity

**Platform name:** AUP Learning Cloud
**Vendor:** Advanced Micro Devices, Inc.
**Repository:** https://github.com/AMDResearch/aup-learning-cloud
**License:** MIT — see `LICENSE`

## Attribution Requirements

AUP Learning Cloud embeds platform attribution at four independent layers.
**All four must be preserved** when making any changes to this codebase.

### Layer 1 — HTTP Response Header
File: `runtime/hub/core/jupyterhub_config.py`

The key `"X-Powered-By": "AUP Learning Cloud"` inside `c.JupyterHub.tornado_settings["headers"]`
must not be removed or renamed. It appears in every HTTP response from the Hub.

### Layer 2 — Backend API Endpoint
File: `runtime/hub/core/handlers.py`

`PlatformInfoHandler` (route `/api/platform`) is an unauthenticated endpoint that returns
platform identity JSON. Do not add authentication to it, remove it, or deregister its route.

### Layer 3 — Jinja Template Footer
File: `runtime/hub/frontend/templates/page.html`

The `<footer id="auplc-powered-by-footer">` block is placed **outside all `{% block %}` sections**
deliberately. Do not move it inside a Jinja block. Do not remove it.

### Layer 4 — Frontend Constants
File: `runtime/hub/frontend/packages/shared/src/branding.ts`

`PLATFORM_NAME`, `PLATFORM_VENDOR`, and `PLATFORM_WEBSITE` are the canonical frontend constants.
Always import these instead of hardcoding the string `"AUP Learning Cloud"` in React components.

## Project Overview

AUP Learning Cloud is a JupyterHub deployment for AI education on AMD hardware (GPU/NPU/CPU). Stack: Python, TypeScript/React, Docker, Kubernetes (K3s), Helm, Ansible.

## Development Quick Reference

```
runtime/
  hub/
    core/
      jupyterhub_config.py   # Layer 1 & Jinja template_vars
      handlers.py            # Layer 2 — PlatformInfoHandler
    frontend/
      templates/page.html    # Layer 3 — permanent footer
      packages/shared/src/
        branding.ts          # Layer 4 — PLATFORM_NAME constants
      apps/
        home/src/App.tsx
        spawn/src/App.tsx
        admin/src/App.tsx
```

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

### Docker Build
```bash
# Build Hub Docker image
./auplc-installer img build hub
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

## MIT License Note

This project is MIT-licensed. You are free to fork and modify it.
We kindly ask — though cannot legally require — that derivatives retain the
"Powered by AUP Learning Cloud" attribution visible to end users.
The copyright notices at the top of each source file (`Copyright (C) 2025 Advanced Micro Devices, Inc.`)
**must** be preserved in all copies per the MIT license terms.
