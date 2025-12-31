# WebShop Workflow
#
# Usage:
#   make setup      # Initialize .env
#   make dev        # Run full stack (UI + Server + Coordinator)
#   make train      # Run training stack (GPU)
#   make logs       # Follow logs
#   make stop       # Stop all services

.PHONY: help setup dev train scale logs stop clean status \
        logs-webshop logs-runner logs-agl logs-ui \
        watch watch-steps watch-progress logs-latest

N ?= 1
LOGS_DIR := logs
LOG_PID_FILE := $(LOGS_DIR)/.log-capture.pid

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
	@echo ""
	@echo "Run:"
	@echo "  make dev         Start Dev Stack (UI + Server + Coordinator)"
	@echo "  make train       Start Training Stack (Server + GPU Coordinator)"
	@echo "  make scale N=3   Scale dev runners to N instances"
	@echo ""
	@echo "Watch (Training Visibility):"
	@echo "  make watch          Launch tmux with 3-pane training view"
	@echo "  make watch-steps    Follow only agent step logs"
	@echo "  make watch-progress Follow only training progress"
	@echo ""
	@echo "Logs:"
	@echo "  make logs        Follow all logs (live)"
	@echo "  make logs-latest Tail the latest log file"
	@echo "  make logs-ui     Follow UI logs"
	@echo "  make logs-runner Follow runner logs"
	@echo "  make logs-agl    Follow coordinator logs"
	@echo "  make logs-webshop Follow WebShop server logs"
	@echo ""
	@echo "Log files are automatically saved to logs/<profile>-<timestamp>.log"
	@echo ""
	@echo "Manage:"
	@echo "  make status      Show container status"
	@echo "  make stop        Stop all services"
	@echo "  make clean       Stop and remove volumes"

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

#==============================================================================
# Run
#==============================================================================

dev: setup ## Start Dev Stack (UI + Server + Coordinator)
	@echo "Starting Dev Stack..."
	@echo "  - UI:       http://localhost:3001"
	@echo "  - WebShop:  http://localhost:3000"
	@echo "  - Store:    http://localhost:4747"
	@echo ""
	@echo "Note: First run downloads ~100MB dataset (takes ~2 min)"
	@echo ""
	docker compose --profile dev up --build -d
	@$(MAKE) --no-print-directory _start-log-capture PROFILE=dev
	@echo ""
	@echo "Logs: $(LOGS_DIR)/dev-$$(ls -t $(LOGS_DIR)/dev-*.log 2>/dev/null | head -1 | xargs -I {} basename {} 2>/dev/null || echo '<timestamp>.log')"
	@echo "Run 'make logs' to follow live or 'make logs-latest' to tail the log file"

train: setup ## Start Training Stack (GPU)
	@echo "Starting Training Stack (GPU)..."
	@echo "  - WebShop:  http://localhost:3000"
	@echo "  - Store:    http://localhost:4747"
	docker compose --profile gpu up --build -d
	@$(MAKE) --no-print-directory _start-log-capture PROFILE=gpu
	@echo ""
	@echo "Logs: $(LOGS_DIR)/gpu-$$(ls -t $(LOGS_DIR)/gpu-*.log 2>/dev/null | head -1 | xargs -I {} basename {} 2>/dev/null || echo '<timestamp>.log')"
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

scale: ## Scale dev runners (e.g., make scale N=3)
	docker compose --profile dev up -d --scale runner=$(N)
	@echo "Scaled runners to $(N) instances"

#==============================================================================
# Watch (Training Visibility)
#==============================================================================

watch: ## Launch tmux session with 3-pane training visibility
	@./scripts/watch-training.sh

watch-steps: ## Follow only agent step logs (compact)
	@docker compose logs -f runner runner-gpu 2>&1 | grep --line-buffered -E '\[TASK\]|\[STEP\]|\[DONE\]'

watch-progress: ## Follow only training progress logs
	@docker compose logs -f agl-server-dev agl-server-gpu 2>&1 | grep --line-buffered -E '\[PROGRESS\]'

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

logs-ui: ## Follow UI logs
	docker compose logs -f ui

logs-runner: ## Follow runner logs
	docker compose logs -f runner runner-gpu

logs-agl: ## Follow coordinator logs
	docker compose logs -f agl-server-dev agl-server-gpu

logs-webshop: ## Follow WebShop server logs
	docker compose logs -f webshop

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
