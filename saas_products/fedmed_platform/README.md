# FedMed Platform

Enterprise-grade federated learning orchestration platform for healthcare institutions.

## Overview

FedMed Platform enables hospitals and healthcare networks to collaboratively train AI models without sharing sensitive patient data. Built on privacy-preserving federated learning, it ensures HIPAA/GDPR compliance while unlocking the power of multi-institutional collaboration.

## Key Features

### Core Capabilities
- **Multi-Institution Federated Training**: Coordinate training across 3-100+ healthcare institutions
- **Real-Time Monitoring**: Track training progress, metrics, and client health
- **Privacy-Preserving**: Data never leaves local institutions
- **HIPAA/GDPR Compliant**: Built-in audit logging and compliance reporting
- **Model Versioning**: Track and manage model checkpoints across training rounds
- **Role-Based Access Control**: Admin, institution admin, researcher, and viewer roles

### Supported Models
- DCGAN (Deep Convolutional GAN) for synthetic medical image generation
- VAE (Variational Autoencoder) - Coming soon
- Diffusion Models - Coming soon

### Key Metrics
- FID (Fréchet Inception Distance) for image quality
- Inception Score (IS) for realism assessment
- Training loss tracking (generator, discriminator)
- Privacy risk assessment

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Web Dashboard (HTMX)                  │
├─────────────────────────────────────────────────┤
│              FastAPI Backend                     │
├─────────────────────────────────────────────────┤
│         Federated Training Engine                │
│  (FedAvg, Model Aggregation, Client Sync)       │
├─────────────────────────────────────────────────┤
│   PostgreSQL    │    Redis    │    Celery       │
└─────────────────────────────────────────────────┘
         │                │              │
    ┌────┴────┐      ┌────┴────┐    ┌───┴────┐
    │ Client1 │      │ Client2 │    │ Client3│
    │Hospital │      │ Clinic  │    │Research│
    └─────────┘      └─────────┘    └────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Installation

#### Option 1: Docker (Recommended)

```bash
cd saas_products/fedmed_platform
cp .env.example .env
# Edit .env with your configuration
docker-compose up -d
```

Access the platform at `http://localhost:8000`

#### Option 2: Manual Setup

```bash
cd saas_products/fedmed_platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Set up database
createdb fedmed_db

# Run migrations (if using Alembic)
alembic upgrade head

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Initial Setup

1. **Create Admin User**:
   - Visit `http://localhost:8000/auth/register`
   - Register with admin credentials
   - Manually update user role to 'admin' in database:
     ```sql
     UPDATE users SET role='admin' WHERE email='admin@example.com';
     ```

2. **Create Institution**:
   - Login as admin
   - Navigate to Institutions → Create New
   - Save API credentials securely

3. **Register Clients**:
   - Navigate to Clients → Create New
   - Assign to institution
   - Configure GPU/resources

4. **Create Training Job**:
   - Navigate to Training → New Job
   - Configure: model type, rounds, epochs, learning rate
   - Assign clients
   - Start training

## Usage

### Creating a Federated Training Job

```python
# Via Web UI: /training/new

# Via API:
curl -X POST http://localhost:8000/api/v1/jobs/create \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Retinal Image Generation",
    "model_type": "dcgan",
    "num_rounds": 10,
    "local_epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "client_ids": ["client_1", "client_2", "client_3"]
  }'
```

### Monitoring Training Progress

```python
# Via Web UI: /training/{job_id}

# Via API:
curl -X GET http://localhost:8000/api/v1/jobs/{job_id} \
  -H "X-API-Key: your-api-key"
```

### Client Heartbeat

```python
# Clients should send heartbeats every 30 seconds:
curl -X POST http://localhost:8000/api/v1/clients/{client_id}/heartbeat \
  -H "X-API-Key: your-api-key"
```

## API Documentation

Full API documentation available at:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | User authentication |
| `/auth/register` | POST | User registration |
| `/training` | GET | List training jobs |
| `/training/create` | POST | Create new training job |
| `/training/{id}` | GET | Get job details |
| `/training/{id}/start` | POST | Start training job |
| `/training/{id}/stop` | POST | Stop training job |
| `/clients` | GET | List clients |
| `/clients/create` | POST | Register new client |
| `/api/v1/jobs` | GET | API: List jobs |
| `/api/v1/clients` | GET | API: List clients |

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key settings:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT signing key (change in production!)
- `MAX_CLIENTS`: Maximum concurrent clients (default: 100)
- `DEFAULT_ROUNDS`: Default federated rounds (default: 10)

### Federated Learning Parameters

```python
# In training job creation:
{
    "num_rounds": 10,           # Federated averaging rounds
    "local_epochs": 5,          # Local training epochs per round
    "batch_size": 16,           # Training batch size
    "learning_rate": 0.0002,    # Learning rate
    "model_type": "dcgan"       # Model architecture
}
```

## Security

### Authentication
- JWT-based authentication
- Secure password hashing with bcrypt
- Token expiration (configurable)

### Authorization
- Role-based access control (RBAC)
- Resource ownership validation
- API key authentication for programmatic access

### Data Protection
- TLS/SSL encryption (configure reverse proxy)
- Database encryption at rest
- Secure API key generation
- HIPAA-compliant audit logging

## Deployment

### Production Deployment

1. **Update Environment Variables**:
   ```bash
   SECRET_KEY=<strong-random-key>
   DEBUG=False
   DATABASE_URL=<production-db>
   ALLOWED_ORIGINS=https://yourdomain.com
   ```

2. **Use HTTPS**:
   - Configure Nginx/Caddy as reverse proxy
   - Obtain SSL certificate (Let's Encrypt)

3. **Scale with Docker Swarm or Kubernetes**:
   - Horizontal scaling for web workers
   - Separate Celery workers for training tasks

4. **Database Backups**:
   - Automated PostgreSQL backups
   - Model checkpoint backups

### Cloud Deployment

#### AWS
- EC2 or ECS for compute
- RDS for PostgreSQL
- ElastiCache for Redis
- S3 for model storage

#### Google Cloud
- Cloud Run or GKE
- Cloud SQL for PostgreSQL
- Memorystore for Redis
- Cloud Storage for models

#### Azure
- App Service or AKS
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Blob Storage for models

## Development

### Running Tests

```bash
pytest tests/ -v --cov=app
```

### Code Quality

```bash
# Format code
black app/

# Lint
flake8 app/

# Type checking (if using mypy)
mypy app/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Troubleshooting

### Common Issues

**Database Connection Error**:
```bash
# Check PostgreSQL is running
pg_isready

# Verify connection string
echo $DATABASE_URL
```

**Redis Connection Error**:
```bash
# Check Redis is running
redis-cli ping

# Should return: PONG
```

**Training Job Stuck**:
- Check Celery worker logs
- Verify client heartbeats
- Check network connectivity

## Monitoring & Observability

### Metrics
- Prometheus metrics at `/metrics` (if enabled)
- Training job success rate
- API response times
- Client connection health

### Logging
- Application logs: `logs/fedmed.log`
- Database query logs (if enabled)
- Audit trail for compliance

## Roadmap

- [ ] Advanced privacy metrics (differential privacy)
- [ ] Multi-GPU support
- [ ] Model deployment automation
- [ ] Integration with PACS systems
- [ ] Mobile client SDK
- [ ] Advanced scheduling (priority queues)
- [ ] Cost tracking and billing

## Support

For issues, feature requests, or questions:
- GitHub Issues: (link to repo)
- Documentation: (link to docs)
- Email: support@fedmed.ai

## License

[Insert License]

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

---

**Built with Python, FastAPI, and HTMX**
