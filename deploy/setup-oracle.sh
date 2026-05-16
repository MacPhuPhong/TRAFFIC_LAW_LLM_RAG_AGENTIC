#!/usr/bin/env bash
set -euo pipefail

# One-shot setup script for Oracle Cloud Always Free (Ubuntu 22.04 ARM64).
# Run as ubuntu user. Idempotent — safe to re-run.

echo "==> 1/6 Updating apt + installing prerequisites..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release git rsync iptables-persistent

echo "==> 2/6 Installing Docker Engine + Compose v2..."
if ! command -v docker >/dev/null 2>&1; then
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo ">>> Please log out + log back in (or run 'newgrp docker') so docker works without sudo."
fi

echo "==> 3/6 Opening port 80 in iptables (Oracle Ubuntu blocks by default)..."
sudo iptables -C INPUT -p tcp -m state --state NEW --dport 80 -j ACCEPT 2>/dev/null || \
  sudo iptables -I INPUT 6 -p tcp -m state --state NEW --dport 80 -j ACCEPT
sudo netfilter-persistent save

echo "==> 4/6 Enabling 2GB swap (safety net, harmless if RAM is plenty)..."
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "==> 5/6 Checking project files..."
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GitHub1}"
if [ ! -d "$PROJECT_ROOT/traffic_rag/deploy" ]; then
  echo "ERROR: Project not found at $PROJECT_ROOT"
  echo "  Upload your repo first (see DEPLOY.md §3). Then re-run this script."
  exit 1
fi

cd "$PROJECT_ROOT/traffic_rag/deploy"
if [ ! -f .env.frontend ]; then
  cp .env.frontend.example .env.frontend
  PUBLIC_IP=$(curl -fsSL https://api.ipify.org || echo "YOUR_SERVER_IP")
  sed -i "s|YOUR_SERVER_IP|$PUBLIC_IP|g" .env.frontend
  RANDOM_SECRET=$(openssl rand -hex 32)
  sed -i "s|CHANGE_ME_random_32_chars|$RANDOM_SECRET|g" .env.frontend
  echo "  Generated .env.frontend with IP=$PUBLIC_IP"
fi
if [ ! -f "$PROJECT_ROOT/.env" ]; then
  echo "ERROR: $PROJECT_ROOT/.env missing (API_KEY, TAVILY_API_KEY, ...)."
  echo "  Upload your local .env and re-run."
  exit 1
fi

echo "==> 6/6 Building + starting containers (first build ~10 min on ARM)..."
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "==> Deploy complete."
PUBLIC_IP=$(curl -fsSL https://api.ipify.org || echo "<server-ip>")
echo "    Frontend:  http://$PUBLIC_IP"
echo "    Logs:      docker compose -f docker-compose.prod.yml logs -f"
echo "    Stop:      docker compose -f docker-compose.prod.yml down"
