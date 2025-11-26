# Makefile for Restaurant Search API

.PHONY: help build up down restart logs shell test clean

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Restaurant Search API - Docker Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# Development
build: ## Build Docker images
	@echo "$(YELLOW)Building Docker images...$(NC)"
	docker-compose build

up: ## Start all services
	@echo "$(YELLOW)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Services started!$(NC)"
	@echo "API:        http://localhost:8080"
	@echo "API Docs:   http://localhost:8080/docs"
	@echo "Health:     http://localhost:8080/health"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000"

up-api: ## Start only API service
	@echo "$(YELLOW)Starting API service...$(NC)"
	docker-compose up -d api
	@echo "$(GREEN)✅ API started at http://localhost:8080$(NC)"

down: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Services stopped$(NC)"

restart: ## Restart all services
	@echo "$(YELLOW)Restarting services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✅ Services restarted$(NC)"

restart-api: ## Restart only API service
	@echo "$(YELLOW)Restarting API...$(NC)"
	docker-compose restart api
	@echo "$(GREEN)✅ API restarted$(NC)"

# Logs
logs: ## View logs from all services
	docker-compose logs -f

logs-api: ## View logs from API service
	docker-compose logs -f api

logs-tail: ## View last 100 lines of logs
	docker-compose logs --tail=100

# Development
shell: ## Open shell in API container
	docker-compose exec api /bin/bash

shell-root: ## Open root shell in API container
	docker-compose exec -u root api /bin/bash

# Testing
test: ## Run tests
	docker-compose exec api pytest tests/

test-coverage: ## Run tests with coverage
	docker-compose exec api pytest --cov=src tests/

health: ## Check API health
	@curl -s http://localhost:8080/health | python -m json.tool

metrics: ## View metrics
	@curl -s http://localhost:8080/metrics | head -n 50

search: ## Test search endpoint (requires query variable)
	@curl -X POST "http://localhost:8080/search" \
		-H "Content-Type: application/json" \
		-d '{"query": "$(query)", "top_k": 3}' | python -m json.tool

# Maintenance
clean: ## Remove all containers, volumes, and images
	@echo "$(YELLOW)Cleaning up...$(NC)"
	docker-compose down -v --rmi all
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

clean-volumes: ## Remove volumes only
	@echo "$(YELLOW)Removing volumes...$(NC)"
	docker-compose down -v
	@echo "$(GREEN)✅ Volumes removed$(NC)"

rebuild: ## Rebuild and restart services
	@echo "$(YELLOW)Rebuilding services...$(NC)"
	docker-compose build --no-cache
	docker-compose up -d
	@echo "$(GREEN)✅ Services rebuilt and restarted$(NC)"

rebuild-api: ## Rebuild and restart API only
	@echo "$(YELLOW)Rebuilding API...$(NC)"
	docker-compose build --no-cache api
	docker-compose up -d api
	@echo "$(GREEN)✅ API rebuilt and restarted$(NC)"

# Status
ps: ## Show running containers
	docker-compose ps

stats: ## Show container resource usage
	docker stats --no-stream

# Backup
backup: ## Backup data directory
	@echo "$(YELLOW)Creating backup...$(NC)"
	@tar -czf backup-$$(date +%Y%m%d-%H%M%S).tar.gz data/
	@echo "$(GREEN)✅ Backup created$(NC)"

# Production
deploy: ## Deploy to production (build and start)
	@echo "$(YELLOW)Deploying to production...$(NC)"
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up -d
	@echo "$(GREEN)✅ Deployed!$(NC)"

# Default target
.DEFAULT_GOAL := help

