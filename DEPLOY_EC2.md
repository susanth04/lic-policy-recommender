# Deploying the LIC Policy Recommender to AWS EC2

This guide walks you through launching and deploying the recommender application on an AWS EC2 instance.

---

## 1. Launch Your AWS EC2 Instance

1. Log into your **AWS Management Console** and navigate to **EC2** → **Launch Instance**.
2. **Name**: `lic-policy-recommender`
3. **OS Image (AMI)**:
   - **Ubuntu Server 24.04 LTS** (Recommended) or **Amazon Linux 2023**.
4. **Instance Type**:
   - `t3.small` (2 vCPU, 2 GB RAM) or `t3.medium` (4 GB RAM) is plenty because the model runs in lightweight `gbr_only` mode (no PyTorch required).
5. **Key Pair**:
   - Select or create a key pair (e.g., `lic-key.pem`) to SSH into the instance.
6. **Network Settings (Security Group)**:
   - Check **Allow SSH traffic** (Port 22 from your IP).
   - Check **Allow HTTP traffic** (Port 80).
   - **Important**: Add a Custom TCP inbound rule for **Port 8000**:
     - **Type**: Custom TCP
     - **Port Range**: `8000`
     - **Source**: `0.0.0.0/0` (Anywhere)
7. Click **Launch Instance**.

---

## 2. Deploy Using Docker Image from Docker Hub (Fastest & Recommended)

The prebuilt image `susanth04/lic-recommender:latest` is published and ready to pull directly onto your EC2 machine.

### Step 1: SSH into your EC2 instance
```bash
ssh -i /path/to/lic-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 2: Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
```

### Step 3: Pull the Prebuilt Image
```bash
docker pull susanth04/lic-recommender:latest
```

### Step 4: Create `.env` configuration file
```bash
cat << 'EOF' > .env
ENABLE_LLM=true
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-lite-latest
LLM_API_KEY=your-gemini-api-key-here
DATABASE_URL=sqlite:////app/db_persist/lic_recommender.db
TOKEN_SECRET=production-secret-token-change-if-desired
EOF
```

### Step 5: Run the Docker Container
```bash
docker run -d \
  --name lic-recommender \
  -p 8000:8000 \
  --env-file .env \
  -v lic_db:/app/db_persist \
  --restart unless-stopped \
  susanth04/lic-recommender:latest
```

### Step 6: Verify Container & Service Health
```bash
docker ps
curl http://localhost:8000/health
```

---

## 3. Alternative Deployment Options

### Option A: Clone Repository & Build Locally with Docker Compose
```bash
git clone https://github.com/<your-repo>/UROP.git
cd UROP
cp .env.example .env
docker compose up -d --build
```

### Option B: Native Systemd Service (Automated Script)

If you prefer running directly in Python on the host:

1. **Run the included setup script**:
   ```bash
   chmod +x deploy/ec2_setup.sh
   ./deploy/ec2_setup.sh
   ```

   This script automatically:
   - Installs system packages and Python 3.11 virtualenv
   - Installs all dependencies
   - Configures `/etc/systemd/system/lic-recommender.service`
   - Enables and starts the background service with automatic restart on reboot.

2. **Manage the Service**:
   ```bash
   # Check logs
   sudo journalctl -u lic-recommender -f

   # Restart service
   sudo systemctl restart lic-recommender
   ```

---

## 4. Access the Live Dashboard

Open your web browser and navigate to:
```
http://<EC2-PUBLIC-IP>:8000/
```

- **Dashboard**: `http://<EC2-PUBLIC-IP>:8000/`
- **Swagger API Docs**: `http://<EC2-PUBLIC-IP>:8000/docs`
- **Health Check**: `http://<EC2-PUBLIC-IP>:8000/health`

---

## 5. User Account & Profile Memory Details

In the **Account** tab of the dashboard, users can now view, add, and update their full personal profile details:

- **Name** (`name`): User's full name
- **Age** (`customer_age`): Entry age in years (used by the eligibility gate)
- **Gender** (`gender`): Male / Female
- **Child's Age** (`child_age`): For junior/child education policies
- **Annual Income** (`annual_income`): In ₹ INR
- **Occupation** (`occupation`): Salaried, Business, Self-Employed, etc.
- **Phone** (`phone`): Contact number
- **Email** (`email`): Contact email address
- **City / Location** (`city`): City and state
- **Preferred Term** (`preferred_term`): Desired policy tenure in years

### Privacy & Reuse Rules:
1. All profile details are saved to the SQLite database (or Postgres) only after the user enables **Profile Memory**.
2. Stored age attributes older than 180 days prompt the user for re-confirmation instead of being silently applied.
3. Users can edit any attribute, download their complete portable JSON (`Download My Data`), or permanently delete their account and history (`Delete All My Data`).
