# Docker Quick Start 🐳

## 🚀 Quick Start (3 Steps)

### 1. Setup Environment

```bash
# Copy example env file
cp env.example .env

# Edit with your API key
nano .env  # or use your favorite editor
```

### 2. Start Services

```bash
# Start API only
docker-compose up -d api

# Or start with monitoring
docker-compose up -d
```

### 3. Test API

```bash
# Check health
curl http://localhost:8080/health

# Try a search
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find Italian restaurants in Dubai", "top_k": 3}'
```

## 📊 Access Services

- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Health**: http://localhost:8080/health
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🛠️ Common Commands

```bash
# View logs
docker-compose logs -f api

# Restart API
docker-compose restart api

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild after code changes
docker-compose build --no-cache api
docker-compose up -d api
```

## 📦 What's Included

- ✅ **API Service** - FastAPI application
- ✅ **Prometheus** - Metrics collection
- ✅ **Grafana** - Visualization dashboards
- ✅ **Health Checks** - Automatic monitoring
- ✅ **Logging** - Centralized logs
- ✅ **Data Persistence** - Volume mounts

## 🔒 Security

- ✅ Non-root user (appuser)
- ✅ Minimal base image (Python slim)
- ✅ Health checks enabled
- ✅ Environment-based secrets
- ✅ Read-only root filesystem capable

## 📝 Configuration

Edit `.env` file:
```bash
OPENAI_API_KEY=your-key-here
ENVIRONMENT=production
PORT=8080
GRAFANA_PASSWORD=your-password
```

## 🐛 Troubleshooting

**API not starting?**
```bash
docker-compose logs api
```

**Port already in use?**
```bash
# Edit docker-compose.yml, change ports:
ports:
  - "8081:8080"  # Use 8081 instead
```

**Need more memory?**
```bash
docker-compose down
# Edit docker-compose.yml, add:
    deploy:
      resources:
        limits:
          memory: 2G
docker-compose up -d
```

## 📚 Full Documentation

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for:
- Detailed configuration
- Production deployment
- Cloud deployment (AWS, GCP, Azure)
- Security best practices
- Performance tuning
- Backup & restore

## 🎯 Next Steps

1. ✅ Access API docs: http://localhost:8080/docs
2. ✅ Try example queries
3. ✅ Setup Grafana dashboards: http://localhost:3000
4. ✅ Monitor metrics: http://localhost:9090

## 🆘 Need Help?

- Check logs: `docker-compose logs -f`
- Health status: `curl http://localhost:8080/health`
- Full guide: [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
- API docs: [docs/API.md](docs/API.md)

---

**Ready!** Your API is running at http://localhost:8080 🎉

