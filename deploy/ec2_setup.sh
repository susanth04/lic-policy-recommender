#!/bin/bash
# ==============================================================================
# Automated Setup Script for AWS EC2 (Ubuntu 22.04 / 24.04 or Amazon Linux 2023)
# ==============================================================================
set -e

echo "=== 1. Updating system packages ==="
if [ -f /etc/debian_version ]; then
    sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv git curl
elif [ -f /etc/amazon-linux-release ] || [ -f /etc/redhat-release ]; then
    sudo dnf update -y && sudo dnf install -y python3.11 python3.11-pip git curl
fi

echo "=== 2. Setting up Python Virtual Environment ==="
python3 -m venv venv || python3.11 -m venv venv
source venv/bin/activate

echo "=== 3. Installing dependencies ==="
pip install --upgrade pip
pip install -r model_out/requirements.txt
pip install python-dotenv sqlalchemy

echo "=== 4. Setting up Systemd Service ==="
APP_DIR=$(pwd)
USER_NAME=$(whoami)

sudo tee /etc/systemd/system/lic-recommender.service > /dev/null <<EOF
[Unit]
Description=LIC Policy Recommender FastAPI Server
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "=== 5. Starting and Enabling lic-recommender service ==="
sudo systemctl daemon-reload
sudo systemctl enable lic-recommender
sudo systemctl restart lic-recommender

echo "=== 6. Status Check ==="
sleep 2
sudo systemctl status lic-recommender --no-pager

echo ""
echo "=========================================================================="
echo "DEPLOYMENT COMPLETE!"
echo "Make sure your AWS EC2 Security Group allows inbound TCP traffic on port 8000."
echo "Access the dashboard at: http://<EC2-PUBLIC-IP>:8000/"
echo "=========================================================================="
