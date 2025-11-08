# MedSynth Platform - Quick Start Guide

Get your MedSynth Platform up and running in 5 minutes!

## Prerequisites

- Docker Desktop installed
- 8GB+ RAM
- 20GB+ free disk space

## Step 1: Environment Setup

Create environment file:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` if needed (defaults work for local development).

## Step 2: Start the Platform

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

Services will start:
- PostgreSQL (database)
- Redis (cache/queue)
- Backend API (FastAPI)
- Celery Worker (training jobs)
- Frontend (React)
- Nginx (reverse proxy)

## Step 3: Access the Platform

Open your browser:

**Main Application:** http://localhost:3000

**API Documentation:** http://localhost:8000/docs

**API Direct:** http://localhost:8000

## Step 4: Create Your Account

1. Go to http://localhost:3000/register
2. Fill in:
   - Full Name
   - Email
   - Password (min 8 characters)
3. Click "Create account"
4. You'll be auto-logged in and redirected to dashboard

## Step 5: Upload a Dataset

1. Click "Upload Dataset" from dashboard or navigate to Datasets
2. Fill in dataset details:
   - **Name:** "My Retinal Dataset"
   - **Type:** Select dataset type (e.g., "Retinal")
   - **Description:** Optional description
3. Upload your data file (TFRecord, images, etc.)
4. Check privacy flags:
   - ✅ Data is anonymized
   - ⬜ Contains PHI (if applicable)
5. Click "Upload Dataset"
6. Wait for processing (status will change to "ready")

## Step 6: Create Training Job

1. Navigate to "Training Jobs" → "New Training Job"
2. Configure:
   - **Name:** "Test Training Run"
   - **Dataset:** Select your uploaded dataset
   - **Clients:** 5 (recommended for testing)
   - **Rounds:** 100
   - **Batch Size:** 32
   - **Learning Rate:** 0.0002
3. Click "Create Training Job"
4. Job will be queued and start automatically

## Step 7: Monitor Progress

1. Click on your training job
2. View real-time progress:
   - Current round / Total rounds
   - Progress percentage
   - Estimated time remaining
3. Check synthetic images every 10 rounds
4. View privacy metrics when complete

## Common Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f celery_worker
docker-compose logs -f frontend
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart backend
docker-compose restart celery_worker
```

### Stop Platform

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

### Database Access

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U medsynth -d medsynth_db

# View tables
\dt

# Exit
\q
```

### Backend Shell

```bash
# Python shell with app context
docker-compose exec backend python

# Execute management commands
docker-compose exec backend python -c "from app.models import init_db; init_db()"
```

## Troubleshooting

### Port Already in Use

If ports 3000, 8000, or 5432 are already in use:

Edit `docker-compose.yml` and change port mappings:

```yaml
ports:
  - "3001:3000"  # Frontend
  - "8001:8000"  # Backend
  - "5433:5432"  # PostgreSQL
```

### Backend Won't Start

Check logs:
```bash
docker-compose logs backend
```

Common issues:
- Database not ready → Wait 10-20 seconds and restart
- Migration errors → Run: `docker-compose exec backend alembic upgrade head`

### Frontend Build Errors

```bash
# Rebuild frontend
docker-compose build frontend
docker-compose up -d frontend
```

### Celery Worker Not Processing Jobs

```bash
# Check worker status
docker-compose logs celery_worker

# Restart worker
docker-compose restart celery_worker

# Check Redis connection
docker-compose exec redis redis-cli ping
```

### Cannot Upload Large Files

Edit `nginx/nginx.conf` and increase `client_max_body_size`:

```nginx
client_max_body_size 50G;  # Increase as needed
```

Then restart:
```bash
docker-compose restart nginx
```

## Sample Data

For testing, you can use sample datasets:

1. **Retinal Images:** Download from Kaggle Diabetic Retinopathy dataset
2. **CT Scans:** Abdominal CT scan datasets
3. **TFRecord Format:** Pre-process images to TFRecord for best performance

## Performance Optimization

### Use GPU for Training

Edit `docker-compose.yml` for celery_worker:

```yaml
celery_worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Increase Worker Concurrency

Edit celery_worker command in `docker-compose.yml`:

```yaml
command: celery -A app.tasks.celery_app worker --concurrency=4 --loglevel=info
```

## Security Notes

**For Production Deployment:**

1. **Change Secret Key:**
   ```bash
   # Generate new secret key
   openssl rand -hex 32
   ```
   Update in `backend/.env`

2. **Use Strong Passwords:**
   - Change database password
   - Use environment-specific passwords

3. **Enable HTTPS:**
   - Configure SSL certificates in Nginx
   - Update CORS origins

4. **Restrict CORS:**
   ```
   BACKEND_CORS_ORIGINS=["https://yourdomain.com"]
   ```

## Next Steps

1. ✅ Platform running
2. ✅ Account created
3. ✅ Dataset uploaded
4. ✅ Training job running

**Now explore:**
- View synthetic images
- Check privacy metrics
- Create multiple training jobs
- Experiment with different parameters
- Read full documentation in `MEDSYNTH_README.md`

## Support

- Documentation: See `MEDSYNTH_README.md`
- Issues: Check logs with `docker-compose logs`
- Community: GitHub Discussions

---

**Congratulations!** 🎉 Your MedSynth Platform is running!
