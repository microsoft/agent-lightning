# WebShop Training Workflow
#
# Usage:
#   make setup       # Initialize .env
#   make dev         # Run dev stack (CPU, external LLM)
#   make train       # Run training stack (GPU)
#   make logs        # Follow logs
#   make stop        # Stop all services
#
# Azure ML:
#   make aml-setup   # One-time AML setup (compute + environment)
#   make aml-train   # Submit training job to Azure ML
#   make aml-logs    # Stream logs from running AML job

.PHONY: help setup build dev train scale logs stop clean status \
        logs-latest watch watch-steps watch-progress \
        aml-setup aml-compute aml-train aml-train-qwen aml-train-dev aml-logs aml-status

N ?= 1
LOGS_DIR := logs
LOG_PID_FILE := $(LOGS_DIR)/.log-capture.pid

# Azure ML defaults (override with environment variables)
AML_RG ?= evalsdemo
AML_WS ?= evalsdemo-ml

.DEFAULT_GOAL := help

#==============================================================================
# Help
#==============================================================================

help: ## Show this help message
	@echo "WebShop Agent - Commands"
	@echo "------------------------"
	@echo ""
	@echo "Setup:"
	@echo "  make setup       Create .env configuration"
	@echo "  make build       Build Docker image"
	@echo ""
	@echo "Run:"
	@echo "  make dev         Start Dev Stack (single container, CPU)"
	@echo "  make train       Start Training Stack (single container, GPU)"
	@echo "  make scale N=3   Set number of runners (default: 1)"
	@echo ""
	@echo "Watch (Training Visibility):"
	@echo "  make watch          Launch tmux with training view"
	@echo "  make watch-steps    Follow agent step logs"
	@echo "  make watch-progress Follow training progress"
	@echo ""
	@echo "Logs:"
	@echo "  make logs        Follow all logs (live)"
	@echo "  make logs-latest Tail the latest log file"
	@echo ""
	@echo "Manage:"
	@echo "  make status      Show container status"
	@echo "  make stop        Stop all services"
	@echo "  make clean       Stop and remove volumes"
	@echo ""
	@echo "Azure ML:"
	@echo "  make aml-setup       One-time AML setup (compute cluster)"
	@echo "  make aml-train       Submit Qwen training job (default)"
	@echo "  make aml-train-qwen  Submit Qwen training job"
	@echo "  make aml-train-dev   Submit dev mode job (external LLM)"
	@echo "  make aml-logs        Stream logs from running AML job"
	@echo "  make aml-status      Show AML job status"

#==============================================================================
# Setup
#==============================================================================

setup: ## Create .env configuration
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example"; \
		echo "Please edit .env and add your OPENAI_API_KEY"; \
	else \
		echo ".env already exists"; \
	fi

build: ## Build Docker image
	@echo "Building WebShop unified image..."
	docker build -f Dockerfile -t webshop-agl ../..
	@echo "Build complete."

build-gpu: ## Build GPU Docker image
	@echo "Building WebShop GPU image..."
	docker build -f Dockerfile --build-arg INSTALL_GPU=true -t webshop-agl-gpu ../..
	@echo "Build complete."

#==============================================================================
# Run
#==============================================================================

dev: setup ## Start Dev Stack (single container, CPU)
	@echo "Starting Dev Stack..."
	@echo "  - WebShop:     http://localhost:3000"
	@echo "  - Coordinator: http://localhost:4747"
	@echo ""
	@echo "Note: First run downloads ~100MB dataset (takes ~2 min)"
	@echo ""
	N_RUNNERS=$(N) docker compose --profile dev up --build -d
	@$(MAKE) --no-print-directory _start-log-capture PROFILE=dev
	@echo ""
	@echo "Run 'make logs' to follow live or 'make logs-latest' to tail the log file"

train: setup ## Start Training Stack (GPU)
	@echo "Starting Training Stack (GPU)..."
	@echo "  - WebShop:     http://localhost:3000"
	@echo "  - Coordinator: http://localhost:4747"
	@echo ""
	N_RUNNERS=$(N) docker compose --profile gpu up --build -d
	@$(MAKE) --no-print-directory _start-log-capture PROFILE=gpu
	@echo ""
	@echo "Run 'make logs' to follow live or 'make logs-latest' to tail the log file"

# Internal target to start background log capture
_start-log-capture:
	@mkdir -p $(LOGS_DIR)
	@# Stop any existing log capture
	@if [ -f $(LOG_PID_FILE) ]; then \
		kill $$(cat $(LOG_PID_FILE)) 2>/dev/null || true; \
		rm -f $(LOG_PID_FILE); \
	fi
	@# Start new log capture in background
	@LOG_FILE=$(LOGS_DIR)/$(PROFILE)-$$(date +%Y%m%d-%H%M%S).log; \
	nohup docker compose --profile $(PROFILE) logs -f > $$LOG_FILE 2>&1 & \
	echo $$! > $(LOG_PID_FILE); \
	echo "Log capture started (PID: $$(cat $(LOG_PID_FILE))) -> $$LOG_FILE"

scale: ## Set number of runners (e.g., make scale N=3)
	@echo "To change runner count, stop and restart with N=$(N):"
	@echo "  make stop && N=$(N) make dev"
	@echo ""
	@echo "Or set N_RUNNERS=$(N) in your .env file"

#==============================================================================
# Watch (Training Visibility)
#==============================================================================

watch: ## Launch tmux session with training visibility
	@./scripts/watch-training.sh

watch-steps: ## Follow agent step logs (compact)
	@docker compose logs -f 2>&1 | grep --line-buffered -E '\[TASK\]|\[STEP\]|\[DONE\]'

watch-progress: ## Follow training progress logs
	@docker compose logs -f 2>&1 | grep --line-buffered -E '\[PROGRESS\]'

#==============================================================================
# Logs
#==============================================================================

logs: ## Follow all logs (live)
	docker compose logs -f

logs-latest: ## Tail the latest log file
	@LATEST=$$(ls -t $(LOGS_DIR)/*.log 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Tailing: $$LATEST"; \
		echo "Press Ctrl+C to stop"; \
		echo ""; \
		tail -f "$$LATEST"; \
	else \
		echo "No log files found in $(LOGS_DIR)/"; \
		echo "Run 'make dev' or 'make train' first"; \
	fi

#==============================================================================
# Manage
#==============================================================================

status: ## Show container status
	docker compose ps -a

stop: ## Stop all services
	@# Stop log capture if running
	@if [ -f $(LOG_PID_FILE) ]; then \
		echo "Stopping log capture (PID: $$(cat $(LOG_PID_FILE)))..."; \
		kill $$(cat $(LOG_PID_FILE)) 2>/dev/null || true; \
		rm -f $(LOG_PID_FILE); \
	fi
	docker compose --profile dev --profile gpu down

clean: ## Stop and remove volumes
	@# Stop log capture if running
	@if [ -f $(LOG_PID_FILE) ]; then \
		kill $$(cat $(LOG_PID_FILE)) 2>/dev/null || true; \
		rm -f $(LOG_PID_FILE); \
	fi
	docker compose --profile dev --profile gpu down -v --rmi local
	@echo "Cleaned up containers, volumes, and local images"
	@echo "Note: Log files in $(LOGS_DIR)/ are preserved"

#==============================================================================
# Azure ML
#==============================================================================

aml-setup: aml-compute ## One-time AML setup (compute cluster)
	@echo ""
	@echo "Azure ML setup complete!"
	@echo "You can now run 'make aml-train' to submit a training job."

aml-compute: ## Create Azure ML compute cluster
	@echo "Creating Azure ML compute cluster..."
	az ml compute create -f aml/compute.yml -g $(AML_RG) -w $(AML_WS) || \
		echo "Compute cluster may already exist (this is OK)"

aml-train: aml-train-qwen ## Submit training job (alias for qwen)

aml-train-qwen: ## Submit Qwen training job
	@echo "Submitting Qwen training job to Azure ML..."
	@echo "Note: Requires HF_TOKEN and WANDB_API_KEY environment variables"
	az ml job create -f aml/jobs/webshop-qwen.yml --stream \
		--set environment_variables.HF_TOKEN="$$HF_TOKEN" \
		--set environment_variables.WANDB_API_KEY="$$WANDB_API_KEY" \
		-g $(AML_RG) -w $(AML_WS)

aml-train-dev: ## Submit dev mode job (external LLM)
	@echo "Submitting dev mode job to Azure ML..."
	@echo "Note: Requires OPENAI_API_KEY environment variable"
	az ml job create -f aml/jobs/webshop-dev.yml --stream \
		--set environment_variables.OPENAI_API_KEY="$$OPENAI_API_KEY" \
		--set environment_variables.OPENAI_API_BASE="$$OPENAI_API_BASE" \
		-g $(AML_RG) -w $(AML_WS)

aml-logs: ## Stream logs from running AML job
	@echo "Streaming logs from most recent job..."
	@JOB_NAME=$$(az ml job list -g $(AML_RG) -w $(AML_WS) --query "[0].name" -o tsv); \
	if [ -n "$$JOB_NAME" ]; then \
		echo "Job: $$JOB_NAME"; \
		az ml job stream -n "$$JOB_NAME" -g $(AML_RG) -w $(AML_WS); \
	else \
		echo "No jobs found"; \
	fi

aml-status: ## Show AML job status
	@echo "Recent Azure ML jobs:"
	az ml job list -g $(AML_RG) -w $(AML_WS) --query "[].{name:name,status:status,created:creation_context.created_at}" -o table
