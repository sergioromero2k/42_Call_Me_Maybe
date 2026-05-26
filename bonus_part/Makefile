# --- Project Name ----
NAME        = call_me_maybe

# --- Executables and Paths ---
PYTHON = uv run python
PIP = uv PIP

# --- Colors for Terminal ---
GREEN        = \033[0;32m
RED          = \033[0;31m
YELLOW       = \033[0;33m

BLUE         = \033[0;34m
PURPLE       = \033[0;35m
CYAN         = \033[0;36m
WHITE        = \033[0;37m

# Bold variants
BOLD_GREEN   = \033[1;32m
BOLD_RED     = \033[1;31m
BOLD_YELLOW  = \033[1;33m
BOLD_BLUE    = \033[1;34m
BOLD_PURPLE  = \033[1;35m
BOLD_CYAN    = \033[1;36m

# Reset color
RESET        = \033[0m

# --- Main rules ---
all : install

install:
	@echo "$(BOLD_CYAN)Installing dependencies with uv...$(RESET)"
	uv sync
	yv add numpy pydantic

# Execute the main script via python interpreter
run:
	@echo "$(GREEN)Running the project...$(RESET)"
	$(PYTHON) -m src

# Execute the script in mode debug with pdb
debug:
	@echo "$(YELLOW)Running in debug mode...$(RESET)"
	$(PYTHON) -m pdb -m src

clean:
	@echo "$(RED)Cleaning up...$(RESET)"
	rm -rf __pycache__ .pytest_cache .mypy_cache .uv [cite: 125, 127]
	find . -type d -name "__pycache__" -exec rm -rf {} +


# --- Linting (Code Quality) ---
lint:
	@echo "$(GREEN)Running Flake8...$(RESET)"
	-$(PYTHON) -m flake8 src/ --max-line-length=79
	@echo "$(GREEN)Running Mypy...$(RESET)"
	-$(PYTHON) -m mypy src/ --no-error-summary

lint-strict:
	@echo "$(RED)Running Flake8...$(RESET)"
	-$(PYTHON) -m flake8 src/ --exclude=venv,test_env,env,.venv

	@echo "$(RED)Running Mypy...$(RESET)"
	-$(PYTHON) -m mypy src/ --strict

.PHONY: all install run debug clean lint lint-strict