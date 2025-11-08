# FedGAN SaaS Products Suite

This directory contains 6 distinct SaaS products built from the FedGAN federated learning framework. Each product targets different market segments and use cases in the healthcare AI space.

## 🏥 Products Overview

### 1. **FedMed Platform** (`fedmed_platform/`)
**Target**: Large hospital networks and healthcare systems
**Purpose**: Enterprise-grade federated learning orchestration platform

**Key Features**:
- Multi-institution federated training management
- Real-time training monitoring dashboard
- Client management and authentication
- Model versioning and deployment
- Compliance reporting (HIPAA/GDPR)
- Role-based access control

**Tech Stack**: FastAPI, HTMX, PostgreSQL, Redis, Celery
**Pricing Model**: Per-institution subscription + compute usage

---

### 2. **SyntheticHealth** (`synthetic_health/`)
**Target**: Medical device companies, research institutions
**Purpose**: On-demand synthetic medical image generation API

**Key Features**:
- REST API for generating synthetic medical images
- Multiple modalities (retinal, CT, X-ray)
- Batch generation with quality controls
- Privacy-preserving image synthesis
- Custom model fine-tuning
- Image download and export

**Tech Stack**: FastAPI, HTMX, SQLite, MinIO/S3
**Pricing Model**: Pay-per-image credits + subscription tiers

---

### 3. **PrivacyGuard** (`privacy_guard/`)
**Target**: Healthcare compliance teams, data protection officers
**Purpose**: Privacy risk assessment and audit platform

**Key Features**:
- Automated privacy vulnerability scanning
- Membership inference attack simulation
- Model inversion risk analysis
- Differential privacy budget calculation
- Compliance audit reports
- Privacy metrics dashboard

**Tech Stack**: FastAPI, HTMX, PostgreSQL, Plotly
**Pricing Model**: Per-model audit + continuous monitoring subscription

---

### 4. **FedTrain Express** (`fedtrain_express/`)
**Target**: Small clinics, research labs, startups
**Purpose**: Simplified federated training-as-a-service

**Key Features**:
- One-click federated training setup
- Pre-configured model templates
- Automated client coordination
- Simple web UI for non-technical users
- Training progress tracking
- Model export and deployment

**Tech Stack**: FastAPI, HTMX, SQLite, Docker
**Pricing Model**: Freemium (3 clients free) + paid tiers

---

### 5. **MedImageQuality** (`medimage_quality/`)
**Target**: Medical imaging vendors, AI researchers
**Purpose**: Automated medical image quality evaluation API

**Key Features**:
- FID (Fréchet Inception Distance) calculation
- Inception Score computation
- Batch image evaluation
- Quality trend analysis
- Comparative benchmarking
- RESTful API + Web UI

**Tech Stack**: FastAPI, HTMX, SQLite, TensorFlow
**Pricing Model**: API calls (pay-per-evaluation) + bulk packages

---

### 6. **DataSim** (`datasim/`)
**Target**: Academic researchers, AI scientists
**Purpose**: Federated data distribution simulator

**Key Features**:
- Non-IID data split generation
- Label skew and feature shift simulation
- Federated scenario benchmarking
- Statistical analysis of data heterogeneity
- Dataset export in multiple formats
- Research-focused analytics

**Tech Stack**: FastAPI, HTMX, SQLite, Pandas
**Pricing Model**: Free for academic + enterprise licensing

---

## 🚀 Quick Start

Each product is self-contained with its own:
- FastAPI backend (`app/main.py`)
- HTMX frontend templates (`templates/`)
- Database models (`models/`)
- API documentation (`/docs`)
- Docker configuration
- Environment configuration (`.env.example`)

### Running Any Product

```bash
cd saas_products/<product_name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
python -m alembic upgrade head

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the application at `http://localhost:8000`

---

## 🏗️ Architecture Overview

All products follow a consistent architecture:

```
product_name/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration and settings
│   ├── database.py          # Database connection and session
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas for validation
│   ├── routers/             # API route handlers
│   ├── services/            # Business logic and core services
│   ├── utils/               # Utility functions
│   └── core/                # Core federated learning components
├── templates/               # Jinja2 + HTMX templates
│   ├── base.html           # Base template with layout
│   ├── components/         # Reusable HTMX components
│   └── pages/              # Full page templates
├── static/
│   ├── css/                # Custom CSS styles
│   ├── js/                 # Minimal JavaScript (Alpine.js)
│   └── images/             # Static images and assets
├── tests/                  # Unit and integration tests
├── alembic/                # Database migrations
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
└── README.md               # Product-specific documentation
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast, async Python web framework
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migrations
- **Pydantic**: Data validation and settings management
- **Celery**: Distributed task queue (for long-running jobs)
- **Redis**: Caching and task broker

### Frontend
- **HTMX**: Dynamic HTML without heavy JavaScript
- **Jinja2**: Template engine
- **Tailwind CSS**: Utility-first CSS framework
- **Alpine.js**: Minimal JavaScript framework for interactivity
- **Chart.js**: Data visualization

### ML/AI
- **TensorFlow**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing

### DevOps
- **Docker**: Containerization
- **PostgreSQL**: Production database
- **SQLite**: Development database
- **Nginx**: Reverse proxy
- **Gunicorn**: WSGI server

---

## 📊 Product Comparison

| Feature | FedMed | SyntheticHealth | PrivacyGuard | FedTrain | MedImageQuality | DataSim |
|---------|--------|----------------|--------------|----------|----------------|---------|
| **Target Users** | Enterprises | Researchers | Compliance | SMBs | Vendors | Academia |
| **Complexity** | High | Medium | Medium | Low | Low | Medium |
| **Price Point** | $$$$$ | $$$ | $$$$ | $$ | $ | Free/$ |
| **Technical Depth** | Expert | Intermediate | Intermediate | Beginner | Beginner | Expert |
| **Federation Support** | Full | No | Audit Only | Simplified | No | Simulation |
| **Privacy Focus** | High | High | Critical | Medium | Low | Research |
| **Scalability** | 100+ clients | N/A | N/A | 10 clients | N/A | N/A |

---

## 🔐 Security & Compliance

All products include:
- ✅ JWT-based authentication
- ✅ Role-based access control (RBAC)
- ✅ Data encryption at rest and in transit
- ✅ HIPAA-compliant logging
- ✅ Audit trail for all operations
- ✅ Rate limiting and DDoS protection
- ✅ SQL injection prevention
- ✅ XSS protection

---

## 📈 Scalability Considerations

### Horizontal Scaling
- All products support multi-instance deployment
- Stateless API design enables load balancing
- Redis for distributed caching
- Celery for distributed task processing

### Database Scaling
- Connection pooling configured
- Read replicas supported
- Sharding strategies documented

### ML Model Serving
- Model caching for fast inference
- Batch prediction optimization
- GPU support for production workloads

---

## 🧪 Testing

Each product includes:
- Unit tests (`pytest`)
- Integration tests
- API endpoint tests
- Load testing scripts (`locust`)
- Security testing guidelines

Run tests:
```bash
cd saas_products/<product_name>
pytest tests/ -v --cov=app
```

---

## 📚 Documentation

Each product has:
- API documentation at `/docs` (Swagger UI)
- Alternative API docs at `/redoc` (ReDoc)
- Admin guide in `docs/admin_guide.md`
- User guide in `docs/user_guide.md`
- Deployment guide in `docs/deployment.md`

---

## 🚢 Deployment

### Development
```bash
uvicorn app.main:app --reload
```

### Production (Docker)
```bash
docker-compose up -d
```

### Cloud Deployment
- AWS: See `docs/deploy_aws.md`
- Google Cloud: See `docs/deploy_gcp.md`
- Azure: See `docs/deploy_azure.md`
- On-premise: See `docs/deploy_on_prem.md`

---

## 🤝 Contributing

Each product is modular and extensible. Contribution guidelines are in each product's README.

---

## 📄 License

Each product inherits the license from the parent FedGAN project.

---

## 🆘 Support

For technical support, please refer to individual product READMEs or contact the development team.

---

**Built with ❤️ using Python, FastAPI, and HTMX**
