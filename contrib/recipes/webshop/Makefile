# WebShop Training Workflow
#
# Usage:
#   make setup       # Initialize .env
#   make train       # Run training stack (GPU)
#   make stop        # Stop all services
#
# Azure ML:
#   make aml-setup   # One-time AML setup (compute + environment)
#   make aml-train   # Submit training job to Azure ML
#   make aml-logs    # Stream logs from running AML job

.PHONY: help setup build build-gpu train scale stop clean status \
        aml-setup aml-compute aml-train aml-train-qwen aml-logs aml-status

N ?= 1

# Azure ML defaults (override with environment variables)
AML_RG ?= <your-resource-group>
AML_WS ?= <your-workspace>


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
	@echo "  make build-gpu   Build GPU Docker image"
	@echo ""
	@echo "Run:"
	@echo "  make train       Start Training Stack (single container, GPU)"
	@echo "  make scale N=3   Set number of runners (default: 1)"
	@echo ""
	@echo "Manage:"
	@echo "  make status      Show container status"
	@echo "  make stop        Stop all services"
	@echo "  make clean       Stop and remove volumes"
	@echo ""
	@echo "Azure ML:"
	@echo "  make aml-train             Submit Qwen training job"
	@echo "  make aml-compute     Create AML compute cluster"
	@echo ""
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

build-gpu: ## Build GPU Docker image
	@echo "Building WebShop GPU image..."
	docker build -f Dockerfile --build-arg INSTALL_GPU=true -t webshop-agl-gpu ../..
	@echo "Build complete."

#==============================================================================
# Run
#==============================================================================

train: setup ## Start Training Stack (GPU)
	@echo "Starting Training Stack (GPU)..."
	@echo "  - WebShop:     http://localhost:3000"
	@echo "  - Coordinator: http://localhost:4747"
	@echo ""
	N_RUNNERS=$(N) docker compose --profile gpu up --build -d

scale: ## Set number of runners (e.g., make scale N=3)
	@echo "To change runner count, stop and restart with N=$(N):"
	@echo "  make stop && N=$(N) make train"
	@echo ""
	@echo "Or set N_RUNNERS=$(N) in your .env file"

#==============================================================================
# Azure ML
#==============================================================================

aml-compute: ## Create Azure ML compute cluster
	@echo "Creating Azure ML compute cluster..."
	az ml compute create -f aml/compute.yml -g $(AML_RG) -w $(AML_WS) || \
		echo "Compute cluster may already exist (this is OK)"


aml-train: ## Submit Qwen training job
	@echo "Submitting Qwen training job to Azure ML..."
	@echo "Note: Requires HF_TOKEN and WANDB_API_KEY environment variables"
	@cp .amlignore ../../../.amlignore && \
	trap 'rm -f ../../../.amlignore' EXIT && \
	az ml job create -f aml/jobs/webshop-qwen.yml --stream \
		--set environment_variables.HF_TOKEN="$$HF_TOKEN" \
		--set environment_variables.WANDB_API_KEY="$$WANDB_API_KEY" \
		-g $(AML_RG) -w $(AML_WS)
