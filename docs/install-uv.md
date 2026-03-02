## Installing **uv** on Ubuntu

### 1. Install from the official script (recommended)
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Developed by Astral.

### 2. Add uv to PATH (if using Zsh / Oh My Zsh)

Edit:
```bash
nano ~/.zshrc
```

Add at the end:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then reload:
```bash
source ~/.zshrc
```

### 3. Verify installation
```bash
uv --version
```

### 4. Quick test
```bash
uv init demo
cd demo
uv run python --version
```