# FedGAN SaaS Products - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FedGAN SaaS Ecosystem                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│  Enterprise  │      │   Research   │     │  Developer   │
│   Segment    │      │   Segment    │     │   Segment    │
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        │                     │                     │
┌───────▼───────────┐ ┌───────▼───────────┐ ┌─────▼─────────┐
│  FedMed Platform  │ │  PrivacyGuard     │ │ MedImageQuality│
│  (Port 8000)      │ │  (Port 8002)      │ │ (Port 8004)    │
│                   │ │                   │ │                │
│ • Multi-hospital  │ │ • Privacy audits  │ │ • FID scores   │
│ • FL orchestration│ │ • Compliance      │ │ • IS scores    │
│ • RBAC            │ │ • Attack sims     │ │ • Batch eval   │
│ • PostgreSQL      │ │ • PostgreSQL      │ │ • RESTful API  │
└───────────────────┘ └───────────────────┘ └────────────────┘

┌───────────────────┐ ┌───────────────────┐ ┌────────────────┐
│ SyntheticHealth   │ │ FedTrain Express  │ │   DataSim      │
│ (Port 8001)       │ │ (Port 8003)       │ │ (Port 8005)    │
│                   │ │                   │ │                │
│ • Image gen API   │ │ • Simplified FL   │ │ • Non-IID sims │
│ • Pay-per-image   │ │ • Pre-built temps │ │ • Dirichlet    │
│ • Credit system   │ │ • Freemium        │ │ • Academic use │
│ • SQLite          │ │ • SQLite          │ │ • Export tools │
└───────────────────┘ └───────────────────┘ └────────────────┘
```

## Technology Stack

### Backend Framework
All products use **FastAPI** for:
- Fast async performance
- Automatic API documentation (Swagger/ReDoc)
- Type safety with Pydantic
- Easy integration with ML frameworks
- WebSocket support for real-time updates

### Frontend
- **HTMX**: Dynamic HTML without heavy JavaScript
- **Tailwind CSS**: Utility-first styling
- **Alpine.js**: Minimal JavaScript interactivity
- **Chart.js**: Data visualizations
- **Jinja2**: Server-side templating

### Databases
- **PostgreSQL**: FedMed Platform, PrivacyGuard (production-grade)
- **SQLite**: SyntheticHealth, FedTrain Express (lightweight)
- **Redis**: Caching and task queues

### ML/AI
- **TensorFlow 2.17**: Deep learning models
- **NumPy**: Numerical computations
- **SciPy**: Statistical analysis
- **Pillow**: Image processing

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Uvicorn**: ASGI server
- **Nginx**: Reverse proxy
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

## Product Comparison Matrix

| Feature | FedMed | SyntheticHealth | PrivacyGuard | FedTrain | MedImageQuality | DataSim |
|---------|--------|----------------|--------------|----------|----------------|---------|
| **Complexity** | High | Medium | Medium | Low | Low | Medium |
| **Target** | Enterprise | Researchers | Compliance | SMBs | Developers | Academia |
| **Database** | PostgreSQL | SQLite | PostgreSQL | SQLite | None | None |
| **Authentication** | Full RBAC | API Key | API Key | Simple | API Key | API Key |
| **Pricing** | $$$$$ | $$$ | $$$$ | $$ | $ | Free/$ |
| **Scalability** | 100+ clients | N/A | N/A | 10 clients | N/A | N/A |
| **Federation** | Full | No | Audit | Simplified | No | Simulation |
| **ML Models** | DCGAN, VAE | DCGAN | Analysis | Templates | Metrics | N/A |
| **Real-time** | Yes | No | No | Yes | No | No |
| **API-First** | Hybrid | Yes | Yes | Hybrid | Yes | Yes |

## Data Flow Architecture

### FedMed Platform - Federated Training Flow

```
┌─────────────┐
│   Web UI    │
└──────┬──────┘
       │ Create Training Job
       ▼
┌─────────────┐
│  FastAPI    │──────┐
│  Backend    │      │
└──────┬──────┘      │
       │             │
       ▼             ▼
┌─────────────┐  ┌──────────┐
│ PostgreSQL  │  │  Celery  │
│  Database   │  │  Worker  │
└─────────────┘  └────┬─────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────┐            ┌──────────────┐
│   Client 1   │            │   Client 2   │
│  (Hospital)  │            │  (Hospital)  │
│              │            │              │
│ • Train Local│            │ • Train Local│
│ • Send Weights            │ • Send Weights
└──────────────┘            └──────────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
                      ▼
              ┌──────────────┐
              │   FedAvg     │
              │ Aggregation  │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │Global Model  │
              │  Updated     │
              └──────────────┘
```

### SyntheticHealth - Image Generation Flow

```
API Request
    │
    ▼
┌──────────────┐
│  FastAPI     │
│  Endpoint    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Check Credits│
│  (Database)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Load DCGAN  │
│  Generator   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Generate    │
│  Image(s)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Save to Disk │
│  or S3       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Deduct Credits│
│Return Image  │
└──────────────┘
```

### PrivacyGuard - Audit Flow

```
Upload Model
    │
    ▼
┌──────────────┐
│  Parse Model │
│  Extract Meta│
└──────┬───────┘
       │
       ├─────────────┬────────────┐
       ▼             ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│Membership│  │  Model   │  │    DP    │
│Inference │  │Inversion │  │ Analysis │
│  Attack  │  │  Attack  │  │          │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┴─────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Aggregate      │
          │ Results        │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │ Generate       │
          │ Report         │
          └────────────────┘
```

## Deployment Architecture

### Single Server Deployment

```
┌─────────────────────────────────────────┐
│           Nginx Reverse Proxy           │
│         (Port 80/443 - HTTPS)           │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┼───────────┬───────────┐
    │           │           │           │
    ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│FedMed  │ │Syntheti│ │Privacy │ │FedTrain│
│:8000   │ │c:8001  │ │G:8002  │ │E:8003  │
└────────┘ └────────┘ └────────┘ └────────┘
    │           │           │           │
    └───────────┼───────────┴───────────┘
                │
    ┌───────────┴───────────┐
    ▼                       ▼
┌────────────┐      ┌──────────────┐
│ PostgreSQL │      │    Redis     │
│  (5432)    │      │   (6379)     │
└────────────┘      └──────────────┘
```

### Multi-Server Production

```
┌─────────────────────────────────────────┐
│         Load Balancer (ALB/NLB)         │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐
│Server 1│ │Server 2│ │Server 3│
│        │ │        │ │        │
│All Apps│ │All Apps│ │All Apps│
└────┬───┘ └────┬───┘ └────┬───┘
     │          │          │
     └──────────┼──────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌─────────────┐      ┌─────────────┐
│RDS/CloudSQL │      │ElastiCache/ │
│ (Managed)   │      │MemoryStore  │
└─────────────┘      └─────────────┘
```

### Kubernetes Deployment

```
┌────────────────────────────────────────┐
│            Ingress Controller          │
│         (nginx-ingress/traefik)        │
└────────────────┬───────────────────────┘
                 │
     ┌───────────┼───────────┬───────────┐
     │           │           │           │
     ▼           ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ FedMed  │ │Synthetic│ │ Privacy │ │FedTrain │
│  Pod    │ │  Pod    │ │  Pod    │ │  Pod    │
│         │ │         │ │         │ │         │
│ Replica │ │ Replica │ │ Replica │ │ Replica │
│ Set (3) │ │ Set (2) │ │ Set (2) │ │ Set (2) │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
     │           │           │           │
     └───────────┴───────────┴───────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
     ▼                       ▼
┌──────────────┐      ┌──────────────┐
│  StatefulSet │      │  StatefulSet │
│  PostgreSQL  │      │    Redis     │
└──────────────┘      └──────────────┘
```

## Security Architecture

### Authentication Flow

```
User/Client
    │
    ▼
┌──────────────┐
│   Login      │
│  (Email/Pwd) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Validate    │
│  Credentials │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Generate    │
│  JWT Token   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Return Token │
│ (HTTPOnly    │
│  Cookie)     │
└──────────────┘
       │
       ▼
All Subsequent Requests
Include Token
```

### Data Encryption

```
┌──────────────────────────────────────┐
│         Data at Rest                 │
│  • Database encryption (AES-256)     │
│  • Model file encryption             │
│  • Backup encryption                 │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│         Data in Transit              │
│  • TLS 1.3 (HTTPS)                   │
│  • Certificate pinning               │
│  • Secure WebSockets (WSS)           │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│         Data in Use                  │
│  • Secure enclaves (optional)        │
│  • Memory encryption                 │
│  • Differential privacy              │
└──────────────────────────────────────┘
```

## Monitoring Architecture

```
┌────────────────────────────────────────┐
│          Application Metrics           │
│  • Request latency                     │
│  • Error rates                         │
│  • Active users                        │
│  • Training job status                 │
└─────────────────┬──────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │   Prometheus   │
         │  (Scraper)     │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │    Grafana     │
         │  (Dashboards)  │
         └────────────────┘
```

## Scalability Considerations

### Horizontal Scaling
- Load balancer distributes traffic
- Stateless application design
- Session data in Redis
- Database read replicas

### Vertical Scaling
- Increase CPU/RAM for ML workloads
- GPU acceleration for image generation
- SSD storage for faster I/O

### Auto-Scaling Triggers
- CPU > 70% for 5 minutes
- Memory > 80% for 5 minutes
- Request queue > 100
- Response time > 2 seconds

---

**Architecture Version**: 1.0
**Last Updated**: 2024
