# =============================================================================
# PCG Embedding Generation Makefile
# =============================================================================
# This Makefile runs multiple encoder scripts to generate embeddings.
# Each encoder script should be named `run_<encoder>.py` (e.g., run_ts2vec.py)
# and output its embeddings to the `embeddings/` directory with a unique name.
# =============================================================================

.PHONY: help install preprocess clean clean-all info \
        ts2vec wav2vec ast fasnet clap \
        all-embeddings

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VENV_NAME      ?= pcg_env
CONDA_NAME     ?= pcg_env
DATA_DIR        = data
RAW_DIR         = $(DATA_DIR)/raw
PROCESSED_DIR   = $(DATA_DIR)/processed
EMBEDDINGS_DIR  = embeddings
MODELS_DIR      = models

# List of available encoder scripts (without the 'run_' prefix)
ENCODERS        = ts2vec wav2vec ast fasnet clap

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "PCG Embedding Generation Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make install         - Set up environment and install dependencies"
	@echo "  make preprocess      - Preprocess raw audio data (must be run first)"
	@echo "  make <encoder>       - Run a specific encoder script, e.g.:"
	@echo "                         make ts2vec"
	@echo "                         make wav2vec"
	@echo "                         make ast"
	@echo "                         make fasnet"
	@echo "                         make clap"
	@echo "  make all-embeddings  - Run all encoders sequentially"
	@echo "  make clean           - Remove generated embeddings and models"
	@echo "  make clean-all       - Remove everything including environment"
	@echo "  make info            - Show system and environment info"
	@echo ""

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
install: check-python create-env install-deps test

check-python:
	@echo "Checking Python version..."
	@python3 -c "import sys; v=sys.version_info; print(f'Python {v.major}.{v.minor}.{v.micro}')"
	@python3 -c "import sys; exit(1) if sys.version_info < (3,8) else exit(0)" || \
		(echo "Error: Python 3.8+ required" && exit 1)

create-env:
	@echo "Creating environment..."
	@if command -v conda >/dev/null 2>&1; then \
		echo "Using conda..."; \
		conda create -n $(CONDA_NAME) python=3.10 -y || true; \
		echo "✓ Conda environment '$(CONDA_NAME)' created"; \
		echo "   Activate with: conda activate $(CONDA_NAME)"; \
	else \
		echo "Using venv..."; \
		python3 -m venv $(VENV_NAME) || echo "Note: Venv may already exist"; \
		echo "✓ Virtual environment '$(VENV_NAME)' created"; \
		echo "   Activate with: source $(VENV_NAME)/bin/activate"; \
	fi

install-deps: create-env
	@echo "Installing dependencies..."
	@$(call run_in_env, \
		pip install --upgrade pip && \
		if [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
			echo "🍎 Apple Silicon: installing MPS PyTorch..."; \
			pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu; \
		elif command -v nvidia-smi >/dev/null 2>&1; then \
			echo "🎮 NVIDIA GPU: installing CUDA PyTorch..."; \
			pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
		else \
			echo "💻 CPU: installing CPU PyTorch..."; \
			pip install torch torchvision torchaudio; \
		fi && \
		pip install numpy scipy librosa tqdm pandas matplotlib scikit-learn soundfile && \
		pip install ts2vec  # Add other encoder packages as needed \
	)
	@echo "✓ All dependencies installed"

test:
	@echo "Testing installation..."
	@$(call run_in_env, python -c "\
import torch, numpy, librosa; \
print('='*50); \
print('TEST RESULTS'); \
print('='*50); \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
if torch.cuda.is_available(): \
    print(f'GPU: {torch.cuda.get_device_name(0)}'); \
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): \
    print('MPS: Available'); \
print(f'NumPy: {numpy.__version__}'); \
print(f'Librosa: {librosa.__version__}'); \
try: import ts2vec; print('TS2Vec: OK'); \
except: print('TS2Vec: Not installed (optional)'); \
print('='*50); \
")

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------
preprocess: dirs
	@echo "Preprocessing raw audio data..."
	@if [ ! -f "preprocess_pcg.py" ]; then \
		echo "Error: preprocess_pcg.py not found!"; exit 1; \
	fi
	@if [ ! -d "$(RAW_DIR)" ] || [ -z "$$(ls -A $(RAW_DIR) 2>/dev/null)" ]; then \
		echo "Warning: $(RAW_DIR) is empty. Place .wav files there."; \
		exit 0; \
	fi
	@$(call run_in_env, python preprocess_pcg.py)
	@echo "✓ Preprocessing complete"

dirs:
	@mkdir -p $(RAW_DIR) $(PROCESSED_DIR) $(EMBEDDINGS_DIR) $(MODELS_DIR)

# -----------------------------------------------------------------------------
# Encoder Targets
# -----------------------------------------------------------------------------
# Each encoder target depends on preprocessing and runs its own script.
# The script should be named run_<encoder>.py (e.g., run_ts2vec.py).
ts2vec wav2vec ast fasnet clap: preprocess
	@echo "Running $@ encoder..."
	@if [ ! -f "run_$@.py" ]; then \
		echo "Error: run_$@.py not found!"; exit 1; \
	fi
	@$(call run_in_env, python run_$@.py)
	@echo "✓ $@ embeddings generated"

all-embeddings: $(ENCODERS)
	@echo "✓ All encoders completed successfully"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:
	@echo "Cleaning generated files..."
	@rm -rf $(EMBEDDINGS_DIR)/*.npy $(EMBEDDINGS_DIR)/*.pkl $(MODELS_DIR)/*
	@rm -f $(PROCESSED_DIR)/*.npy   # optional: keep processed data?
	@echo "✓ Cleaned embeddings and models"

clean-all: clean
	@echo "Removing environment..."
	@if command -v conda >/dev/null 2>&1; then \
		conda env remove -n $(CONDA_NAME) -y 2>/dev/null || true; \
		echo "✓ Conda environment removed"; \
	else \
		rm -rf $(VENV_NAME); \
		echo "✓ Virtual environment removed"; \
	fi
	@echo "✓ Full cleanup done"

# -----------------------------------------------------------------------------
# Info
# -----------------------------------------------------------------------------
info:
	@echo "System Information:"
	@echo "  OS: $$(uname -s)"
	@echo "  Arch: $$(uname -m)"
	@echo "  Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@if command -v conda >/dev/null 2>&1; then \
		echo "  Conda: Available"; \
	else \
		echo "  Conda: Not available (using venv)"; \
	fi
	@echo ""
	@echo "Project Structure:"
	@echo "  Raw data:      $(RAW_DIR)"
	@echo "  Processed:     $(PROCESSED_DIR)"
	@echo "  Embeddings:    $(EMBEDDINGS_DIR)"
	@echo "  Models:        $(MODELS_DIR)"
	@echo ""
	@echo "Available encoders: $(ENCODERS)"

# -----------------------------------------------------------------------------
# Helper: run a command inside the active environment
# -----------------------------------------------------------------------------
define run_in_env
	@if command -v conda >/dev/null 2>&1; then \
		. "$$(conda info --base)/etc/profile.d/conda.sh" && conda activate $(CONDA_NAME) && $(1) ; \
	else \
		. ./$(VENV_NAME)/bin/activate && $(1) ; \
	fi
endef