#!/usr/bin/env bash
# Setup script for the Bloomberg flight-delay case study.
# Run once:  bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"
echo "Working in: $(pwd)"
echo

# ---------------------------------------------------------------- 1. Python
echo "==> Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found."
  echo "Install it with:  brew install python@3.12"
  echo "(If you don't have Homebrew: https://brew.sh)"
  exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "    Found Python $PY_VER at $(command -v python3)"

# Need 3.9+; 3.10+ strongly preferred for current pandas/sklearn wheels.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)'; then
  echo "ERROR: Python 3.9+ required, found $PY_VER."
  echo "Install a newer one with:  brew install python@3.12"
  exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "    WARNING: Python $PY_VER works but 3.10+ is recommended."
fi
echo

# ---------------------------------------------------------- 2. Virtual env
if [ -d ".venv" ]; then
  echo "==> Virtual environment already exists, reusing it."
else
  echo "==> Creating virtual environment in .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "    Active: $(command -v python)"
echo

# -------------------------------------------------------------- 3. libomp
# XGBoost's macOS wheel links against the OpenMP runtime, which Apple does not
# ship. Without it, `import xgboost` fails with a dyld library error.
echo "==> Checking OpenMP runtime (needed by XGBoost on macOS)..."
if [[ "$OSTYPE" == darwin* ]]; then
  if command -v brew >/dev/null 2>&1; then
    if brew list libomp >/dev/null 2>&1; then
      echo "    libomp already installed."
    else
      echo "    Installing libomp via Homebrew..."
      brew install libomp || {
        echo "    WARNING: libomp install failed. XGBoost may not import."
        echo "    Fallback: use sklearn's HistGradientBoostingClassifier instead."
      }
    fi
  else
    echo "    WARNING: Homebrew not found, so libomp cannot be installed."
    echo "    XGBoost may fail to import. Either install Homebrew"
    echo "    (https://brew.sh) then run 'brew install libomp', or fall back to"
    echo "    sklearn's HistGradientBoostingClassifier."
  fi
else
  echo "    Not macOS - skipping."
fi
echo

# ------------------------------------------------------------ 4. Packages
echo "==> Upgrading pip..."
python -m pip install --quiet --upgrade pip setuptools wheel

echo "==> Installing packages (this takes a few minutes)..."
python -m pip install --quiet -r requirements.txt
echo "    Done."
echo

# -------------------------------------------------------- 5. Jupyter kernel
echo "==> Registering Jupyter kernel..."
python -m ipykernel install --user \
  --name bloomberg-case \
  --display-name "Python (bloomberg-case)" >/dev/null 2>&1
echo "    Kernel 'Python (bloomberg-case)' registered."
echo

# ------------------------------------------------------------ 6. Jupytext
# Pairs every .ipynb with a .py in percent format, so notebooks get clean
# git diffs and can be edited as plain text.
echo "==> Configuring jupytext pairing..."
cat > jupytext.toml <<'EOF'
# Pair every notebook with a percent-format .py script.
# Editing either side and saving syncs the other.
formats = "ipynb,py:percent"
EOF
echo "    Wrote jupytext.toml"
echo

# ---------------------------------------------------------- 7. Directories
echo "==> Creating project directories..."
mkdir -p notebooks src data/raw data/interim data/sample figures
touch src/__init__.py
echo "    notebooks/ src/ data/{raw,interim,sample}/ figures/"
echo

# -------------------------------------------------------------- 8. Verify
echo "==> Verifying installation..."
python - <<'EOF'
import importlib, sys
pkgs = ["jupyterlab", "jupytext", "pandas", "numpy", "pyarrow",
        "matplotlib", "seaborn", "sklearn", "xgboost", "statsmodels",
        "shap", "requests", "tqdm", "holidays"]
failed = []
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"    OK   {p:14s} {getattr(m, '__version__', '')}")
    except Exception:
        failed.append(p)
        print(f"    FAIL {p}")
if failed:
    print("\nSome packages failed to import:", ", ".join(failed))
    sys.exit(1)
print("\nAll packages imported successfully.")
EOF
echo

echo "======================================================================"
echo " Setup complete."
echo
echo " To start working:"
echo "     cd $(pwd)"
echo "     source .venv/bin/activate"
echo "     jupyter lab"
echo
echo " In JupyterLab, pick the 'Python (bloomberg-case)' kernel."
echo "======================================================================"
