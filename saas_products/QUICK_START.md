# Quick Start Guide - FedGAN SaaS Products

Get all 6 SaaS products running in under 5 minutes!

## One-Command Setup (All Products)

```bash
cd saas_products

# Install all dependencies
for product in fedmed_platform synthetic_health privacy_guard fedtrain_express medimage_quality datasim; do
    cd $product
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    cd ..
done

echo "✅ All products configured!"
```

## Start Individual Products

### 1. FedMed Platform (Port 8000)
**Enterprise Federated Learning Platform**

```bash
cd fedmed_platform
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Test**: Open http://localhost:8000
- Landing page with enterprise features
- Login/Register functionality
- Dashboard for training jobs

---

### 2. SyntheticHealth (Port 8001)
**Synthetic Medical Image Generation API**

```bash
cd synthetic_health
source venv/bin/activate
uvicorn app.main:app --reload --port 8001
```

**Test API**:
```bash
curl http://localhost:8001/health
```

---

### 3. PrivacyGuard (Port 8002)
**Privacy Risk Assessment Dashboard**

```bash
cd privacy_guard
source venv/bin/activate
uvicorn app.main:app --reload --port 8002
```

**Test**:
```bash
curl http://localhost:8002
```

---

### 4. FedTrain Express (Port 8003)
**Simplified Federated Training**

```bash
cd fedtrain_express
source venv/bin/activate
uvicorn app.main:app --reload --port 8003
```

**Test**:
```bash
curl http://localhost:8003/templates
```

---

### 5. MedImageQuality (Port 8004)
**Image Quality Evaluation API**

```bash
cd medimage_quality
source venv/bin/activate
uvicorn app.main:app --reload --port 8004
```

**Test API**:
```bash
curl -X POST http://localhost:8004/api/v1/fid \
  -F "real_images=@test.zip" \
  -F "generated_images=@test.zip"
```

---

### 6. DataSim (Port 8005)
**Non-IID Data Distribution Simulator**

```bash
cd datasim
source venv/bin/activate
uvicorn app.main:app --reload --port 8005
```

**Test API**:
```bash
curl -X POST http://localhost:8005/api/v1/split \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "retinopathy",
    "num_clients": 3,
    "strategy": "dirichlet",
    "alpha": 0.5
  }'
```

---

## Run All Products at Once

**Option 1: Using tmux**

```bash
# Install tmux
sudo apt install tmux

# Start session
tmux new -s fedgan

# Split into 6 panes
Ctrl+B then "  # Split horizontal
Ctrl+B then %  # Split vertical
# Repeat to create 6 panes

# In each pane, run a product
cd fedmed_platform && uvicorn app.main:app --port 8000
cd synthetic_health && uvicorn app.main:app --port 8001
cd privacy_guard && uvicorn app.main:app --port 8002
cd fedtrain_express && uvicorn app.main:app --port 8003
cd medimage_quality && uvicorn app.main:app --port 8004
cd datasim && uvicorn app.main:app --port 8005
```

**Option 2: Using Docker Compose**

```bash
# Coming soon: docker-compose-all.yml
docker-compose -f docker-compose-all.yml up -d
```

---

## Access All Products

Once running:

| Product | URL | API Docs |
|---------|-----|----------|
| FedMed Platform | http://localhost:8000 | http://localhost:8000/api/docs |
| SyntheticHealth | http://localhost:8001 | http://localhost:8001/api/docs |
| PrivacyGuard | http://localhost:8002 | http://localhost:8002/api/docs |
| FedTrain Express | http://localhost:8003 | http://localhost:8003/api/docs |
| MedImageQuality | http://localhost:8004 | http://localhost:8004/api/docs |
| DataSim | http://localhost:8005 | http://localhost:8005/api/docs |

---

## Next Steps

1. **Read Individual READMEs**: Each product has detailed documentation
2. **Configure Databases**: Setup PostgreSQL for production use
3. **Explore API Docs**: Visit `/api/docs` for interactive API documentation
4. **Deploy to Production**: See `DEPLOYMENT_GUIDE.md`
5. **Integrate with FedGAN**: Connect to existing FedGAN models

---

## Troubleshooting

**Port Already in Use**:
```bash
# Kill process on port 8000
kill -9 $(lsof -t -i:8000)
```

**Module Not Found**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Database Connection Error**:
```bash
# For FedMed Platform, ensure PostgreSQL is running
sudo systemctl start postgresql
createdb fedmed_db
```

---

## Demo Credentials

**FedMed Platform**:
- After registration, manually update role to 'admin' in database
- Email: admin@example.com
- Password: (your chosen password)

---

## Support

- Full Documentation: See `README.md` in each product directory
- Deployment Guide: `DEPLOYMENT_GUIDE.md`
- GitHub Issues: [Report bugs]

**Happy Building! 🚀**
