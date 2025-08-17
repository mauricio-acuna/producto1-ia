# Guía de Despliegue en Producción

Patrones de despliegue de nivel empresarial para el Framework de Agentes IA en entornos de producción.

## Tabla de Contenidos

- [Resumen General](#resumen-general)
- [Patrones de Arquitectura](#patrones-de-arquitectura)
- [Requisitos de Infraestructura](#requisitos-de-infraestructura)
- [Despliegue con Contenedores](#despliegue-con-contenedores)
- [Despliegue en Kubernetes](#despliegue-en-kubernetes)
- [Despliegues en Proveedores de Nube](#despliegues-en-proveedores-de-nube)
- [Configuración de Seguridad](#configuración-de-seguridad)
- [Monitoreo y Observabilidad](#monitoreo-y-observabilidad)
- [Optimización de Rendimiento](#optimización-de-rendimiento)
- [Recuperación ante Desastres](#recuperación-ante-desastres)
- [Estrategias de Escalamiento](#estrategias-de-escalamiento)
- [Solución de Problemas](#solución-de-problemas)

---

## Resumen General

Esta guía proporciona patrones de despliegue de nivel empresarial para el Framework de Agentes IA, siguiendo las mejores prácticas de la industria de organizaciones como OpenAI, Google, Microsoft y Anthropic.

### Deployment Objectives

- **High Availability**: 99.9% uptime SLA
- **Scalability**: Handle 10,000+ concurrent requests
- **Security**: Zero-trust architecture with compliance
- **Performance**: Sub-2s response times at P95
- **Observability**: Complete monitoring and alerting
- **Compliance**: SOC2, GDPR, HIPAA ready

### Reference Architecture

```mermaid
graph TB
    subgraph "External"
        U[Users] --> LB[Load Balancer]
        API[External APIs] --> AG[API Gateway]
    end
    
    subgraph "Edge Layer"
        LB --> CDN[CDN/Edge Cache]
        CDN --> WAF[Web Application Firewall]
        WAF --> AG
    end
    
    subgraph "Application Layer"
        AG --> MS1[Agent Service 1]
        AG --> MS2[Agent Service 2] 
        AG --> MS3[Agent Service N]
        
        MS1 --> MQ[Message Queue]
        MS2 --> MQ
        MS3 --> MQ
    end
    
    subgraph "Data Layer"
        MS1 --> DB[(Primary Database)]
        MS2 --> DB
        MS3 --> DB
        
        MS1 --> VDB[(Vector Database)]
        MS2 --> VDB
        MS3 --> VDB
        
        MS1 --> CACHE[(Redis Cache)]
        MS2 --> CACHE
        MS3 --> CACHE
    end
    
    subgraph "Infrastructure"
        MON[Monitoring Stack]
        LOG[Logging Stack]
        SEC[Security Stack]
    end
```

---

## Architecture Patterns

### Microservices Architecture

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    image: ai-agent-framework/gateway:1.4.2
    ports:
      - "443:443"
      - "80:80"
    environment:
      - ENVIRONMENT=production
      - SSL_CERT_PATH=/certs/server.crt
      - SSL_KEY_PATH=/certs/server.key
    volumes:
      - ./certs:/certs:ro
      - ./config/gateway.yml:/etc/gateway/gateway.yml:ro
    depends_on:
      - agent-orchestrator
      - tool-registry
      - knowledge-service
    networks:
      - ai-agent-network
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        max_attempts: 3

  # Agent Orchestrator
  agent-orchestrator:
    image: ai-agent-framework/orchestrator:1.4.2
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET=${JWT_SECRET}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
    volumes:
      - ./config/orchestrator.yml:/etc/orchestrator/config.yml:ro
    depends_on:
      - postgres
      - redis
      - message-queue
    networks:
      - ai-agent-network
    deploy:
      replicas: 5
      resources:
        limits:
          memory: 2G
          cpus: '1'
      restart_policy:
        condition: on-failure

  # Tool Registry Service
  tool-registry:
    image: ai-agent-framework/tool-registry:1.4.2
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    networks:
      - ai-agent-network
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Knowledge Service (RAG)
  knowledge-service:
    image: ai-agent-framework/knowledge:1.4.2
    environment:
      - ENVIRONMENT=production
      - VECTOR_DB_URL=${VECTOR_DB_URL}
      - EMBEDDING_SERVICE_URL=${EMBEDDING_SERVICE_URL}
    volumes:
      - knowledge-data:/data
    networks:
      - ai-agent-network
    deploy:
      replicas: 4
      resources:
        limits:
          memory: 4G
          cpus: '2'

  # Database Services
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=ai_agent_framework
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    networks:
      - ai-agent-network
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    networks:
      - ai-agent-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  # Vector Database (Weaviate)
  weaviate:
    image: semitechnologies/weaviate:1.21.2
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=false
      - AUTHENTICATION_OIDC_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    volumes:
      - weaviate-data:/var/lib/weaviate
    networks:
      - ai-agent-network
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'

  # Message Queue (RabbitMQ)
  message-queue:
    image: rabbitmq:3.12-management-alpine
    environment:
      - RABBITMQ_DEFAULT_USER=${MQ_USER}
      - RABBITMQ_DEFAULT_PASS=${MQ_PASSWORD}
    volumes:
      - rabbitmq-data:/var/lib/rabbitmq
    networks:
      - ai-agent-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:v2.45.0
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - ai-agent-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  grafana:
    image: grafana/grafana:10.0.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    networks:
      - ai-agent-network
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Logging Stack
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.9.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - ai-agent-network
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

  logstash:
    image: docker.elastic.co/logstash/logstash:8.9.0
    volumes:
      - ./config/logstash/pipeline:/usr/share/logstash/pipeline:ro
    networks:
      - ai-agent-network
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.9.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - ai-agent-network
    depends_on:
      - elasticsearch

volumes:
  postgres-data:
  redis-data:
  weaviate-data:
  rabbitmq-data:
  knowledge-data:
  prometheus-data:
  grafana-data:
  elasticsearch-data:

networks:
  ai-agent-network:
    driver: overlay
    attachable: true
```

---

## Infrastructure Requirements

### Minimum Production Requirements

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| **API Gateway** | 2 cores | 4GB | 20GB SSD | 1Gbps |
| **Agent Orchestrator** | 4 cores | 8GB | 50GB SSD | 1Gbps |
| **Tool Registry** | 2 cores | 4GB | 20GB SSD | 1Gbps |
| **Knowledge Service** | 8 cores | 16GB | 100GB SSD | 1Gbps |
| **Database (PostgreSQL)** | 4 cores | 16GB | 500GB SSD | 1Gbps |
| **Vector DB (Weaviate)** | 8 cores | 32GB | 1TB SSD | 1Gbps |
| **Cache (Redis)** | 2 cores | 8GB | 50GB SSD | 1Gbps |
| **Message Queue** | 2 cores | 4GB | 50GB SSD | 1Gbps |
| **Monitoring Stack** | 4 cores | 8GB | 200GB SSD | 1Gbps |

### Recommended Enterprise Requirements

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| **API Gateway** | 4 cores | 8GB | 50GB SSD | 10Gbps |
| **Agent Orchestrator** | 8 cores | 16GB | 100GB SSD | 10Gbps |
| **Tool Registry** | 4 cores | 8GB | 50GB SSD | 10Gbps |
| **Knowledge Service** | 16 cores | 32GB | 500GB SSD | 10Gbps |
| **Database (PostgreSQL)** | 8 cores | 32GB | 2TB SSD | 10Gbps |
| **Vector DB (Weaviate)** | 16 cores | 64GB | 5TB SSD | 10Gbps |
| **Cache (Redis)** | 4 cores | 16GB | 100GB SSD | 10Gbps |
| **Message Queue** | 4 cores | 8GB | 100GB SSD | 10Gbps |
| **Monitoring Stack** | 8 cores | 16GB | 1TB SSD | 10Gbps |

---

## Container Deployment

### Production Dockerfile

```dockerfile
# Multi-stage production Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY src/ ./src/

# Build application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

WORKDIR /app

# Copy built application
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./package.json

# Security: Remove unnecessary packages
RUN apk del --purge && \
    rm -rf /var/cache/apk/*

# Set security headers
ENV NODE_ENV=production
ENV NODE_OPTIONS="--max-old-space-size=2048"

# Switch to non-root user
USER nodejs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node dist/health-check.js

# Expose port
EXPOSE 3000

# Start application
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]
```

### Build and Security Scanning

```bash
#!/bin/bash
# build-and-scan.sh

set -euo pipefail

# Build image
docker build -t ai-agent-framework/orchestrator:latest .

# Security scanning with Trivy
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):/workspace \
    aquasec/trivy:latest image \
    --severity HIGH,CRITICAL \
    --no-progress \
    --format json \
    --output /workspace/security-scan.json \
    ai-agent-framework/orchestrator:latest

# Check for vulnerabilities
if [ -s security-scan.json ]; then
    echo "Security vulnerabilities found. Review security-scan.json"
    exit 1
fi

# Tag for production
docker tag ai-agent-framework/orchestrator:latest \
    ai-agent-framework/orchestrator:$(git rev-parse --short HEAD)

echo "Build and security scan completed successfully"
```

---

## Kubernetes Deployment

### Namespace and RBAC

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-agent-framework
  labels:
    name: ai-agent-framework
    environment: production

---
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ai-agent-framework-sa
  namespace: ai-agent-framework

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ai-agent-framework-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ai-agent-framework-binding
subjects:
- kind: ServiceAccount
  name: ai-agent-framework-sa
  namespace: ai-agent-framework
roleRef:
  kind: ClusterRole
  name: ai-agent-framework-role
  apiGroup: rbac.authorization.k8s.io
```

### Agent Orchestrator Deployment

```yaml
# agent-orchestrator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  namespace: ai-agent-framework
  labels:
    app: agent-orchestrator
    version: v1.4.2
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    metadata:
      labels:
        app: agent-orchestrator
        version: v1.4.2
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "3000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: ai-agent-framework-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: agent-orchestrator
        image: ai-agent-framework/orchestrator:1.4.2
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 3000
          name: http
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: jwt-secret
              key: secret
        envFrom:
        - configMapRef:
            name: agent-orchestrator-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
          readOnly: true
        - name: temp-volume
          mountPath: /tmp
      volumes:
      - name: config-volume
        configMap:
          name: agent-orchestrator-config
      - name: temp-volume
        emptyDir: {}
      nodeSelector:
        kubernetes.io/arch: amd64
      tolerations:
      - key: "ai-workload"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"

---
apiVersion: v1
kind: Service
metadata:
  name: agent-orchestrator-service
  namespace: ai-agent-framework
  labels:
    app: agent-orchestrator
spec:
  selector:
    app: agent-orchestrator
  ports:
  - port: 80
    targetPort: 3000
    protocol: TCP
    name: http
  type: ClusterIP

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-orchestrator-config
  namespace: ai-agent-framework
data:
  LOG_LEVEL: "info"
  MAX_CONCURRENT_EXECUTIONS: "100"
  EXECUTION_TIMEOUT: "300"
  METRICS_ENABLED: "true"
  TRACING_ENABLED: "true"
  JAEGER_ENDPOINT: "http://jaeger-collector:14268/api/traces"
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-orchestrator-hpa
  namespace: ai-agent-framework
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-orchestrator
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: agent_executions_per_second
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

### Network Policies

```yaml
# network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-orchestrator-netpol
  namespace: ai-agent-framework
spec:
  podSelector:
    matchLabels:
      app: agent-orchestrator
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 3000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

---

## Cloud Provider Deployments

### AWS EKS Deployment

```yaml
# aws-eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ai-agent-framework-cluster
  region: us-west-2
  version: "1.27"

iam:
  withOIDC: true
  serviceAccounts:
  - metadata:
      name: ai-agent-framework-sa
      namespace: ai-agent-framework
    roleOnly: true
    attachPolicyARNs:
    - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
    - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

vpc:
  enableDnsHostnames: true
  enableDnsSupport: true
  cidr: '192.168.0.0/16'
  nat:
    gateway: HighlyAvailable

nodeGroups:
- name: ai-agent-nodes
  instanceType: c5.2xlarge
  minSize: 3
  maxSize: 20
  desiredCapacity: 5
  availabilityZones: ["us-west-2a", "us-west-2b", "us-west-2c"]
  volumeSize: 100
  volumeType: gp3
  amiFamily: AmazonLinux2
  iam:
    attachPolicyARNs:
    - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
    - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
    - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
    - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
  labels:
    nodegroup-type: ai-agent-nodes
  taints:
  - key: ai-workload
    value: "true"
    effect: NoSchedule
  tags:
    Environment: production
    Project: ai-agent-framework

addons:
- name: vpc-cni
  version: latest
- name: coredns
  version: latest
- name: kube-proxy
  version: latest
- name: aws-ebs-csi-driver
  version: latest

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]
    logRetentionInDays: 30
```

### AWS RDS Configuration

```yaml
# aws-rds.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rds-credentials
  namespace: ai-agent-framework
type: Opaque
stringData:
  username: admin
  password: ${RDS_PASSWORD}
  endpoint: ai-agent-db.cluster-xyz.us-west-2.rds.amazonaws.com

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rds-config
  namespace: ai-agent-framework
data:
  DATABASE_URL: "postgresql://admin:${RDS_PASSWORD}@ai-agent-db.cluster-xyz.us-west-2.rds.amazonaws.com:5432/ai_agent_framework?sslmode=require"
```

### Azure AKS Deployment

```yaml
# azure-aks-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aks-deployment-config
data:
  deploy.sh: |
    #!/bin/bash
    
    # Create resource group
    az group create --name ai-agent-framework-rg --location eastus
    
    # Create AKS cluster
    az aks create \
      --resource-group ai-agent-framework-rg \
      --name ai-agent-framework-cluster \
      --node-count 3 \
      --node-vm-size Standard_D4s_v3 \
      --enable-addons monitoring \
      --enable-cluster-autoscaler \
      --min-count 1 \
      --max-count 20 \
      --network-plugin azure \
      --network-policy azure \
      --kubernetes-version 1.27.3 \
      --zones 1 2 3
    
    # Get credentials
    az aks get-credentials \
      --resource-group ai-agent-framework-rg \
      --name ai-agent-framework-cluster
    
    # Create Azure Container Registry
    az acr create \
      --resource-group ai-agent-framework-rg \
      --name aiagentframeworkacr \
      --sku Premium
    
    # Attach ACR to AKS
    az aks update \
      --resource-group ai-agent-framework-rg \
      --name ai-agent-framework-cluster \
      --attach-acr aiagentframeworkacr
```

### Google GKE Deployment

```yaml
# gcp-gke-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gke-deployment-config
data:
  deploy.sh: |
    #!/bin/bash
    
    # Set project and zone
    export PROJECT_ID=ai-agent-framework-project
    export ZONE=us-central1-a
    export CLUSTER_NAME=ai-agent-framework-cluster
    
    # Create GKE cluster
    gcloud container clusters create $CLUSTER_NAME \
      --project=$PROJECT_ID \
      --zone=$ZONE \
      --machine-type=n2-standard-4 \
      --num-nodes=3 \
      --min-nodes=1 \
      --max-nodes=20 \
      --enable-autoscaling \
      --enable-autorepair \
      --enable-autoupgrade \
      --enable-network-policy \
      --enable-cloud-logging \
      --enable-cloud-monitoring \
      --disk-size=100GB \
      --disk-type=pd-ssd \
      --image-type=COS_CONTAINERD \
      --cluster-version=1.27.3-gke.100
    
    # Get credentials
    gcloud container clusters get-credentials $CLUSTER_NAME \
      --zone=$ZONE \
      --project=$PROJECT_ID
    
    # Create node pool for AI workloads
    gcloud container node-pools create ai-workload-pool \
      --cluster=$CLUSTER_NAME \
      --zone=$ZONE \
      --machine-type=n2-highmem-8 \
      --num-nodes=2 \
      --min-nodes=0 \
      --max-nodes=10 \
      --enable-autoscaling \
      --node-taints=ai-workload=true:NoSchedule
```

---

## Security Configuration

### TLS/SSL Configuration

```yaml
# tls-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
  namespace: ai-agent-framework
type: kubernetes.io/tls
data:
  tls.crt: ${TLS_CERT_BASE64}
  tls.key: ${TLS_KEY_BASE64}

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-agent-framework-ingress
  namespace: ai-agent-framework
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
    nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-RSA-AES128-GCM-SHA256,ECDHE-RSA-AES256-GCM-SHA384"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.ai-agent-framework.com
    secretName: tls-secret
  rules:
  - host: api.ai-agent-framework.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway-service
            port:
              number: 80
```

### Security Context and Pod Security Standards

```yaml
# security-context.yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    runAsGroup: 1001
    fsGroup: 1001
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: agent-orchestrator
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 1001
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp-volume
      mountPath: /tmp
    - name: var-cache-volume
      mountPath: /var/cache
  volumes:
  - name: tmp-volume
    emptyDir: {}
  - name: var-cache-volume
    emptyDir: {}
```

### Secrets Management with External Secrets Operator

```yaml
# external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-secret-store
  namespace: ai-agent-framework
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "ai-agent-framework"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-secret
  namespace: ai-agent-framework
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-secret-store
    kind: SecretStore
  target:
    name: database-secret
    creationPolicy: Owner
  data:
  - secretKey: url
    remoteRef:
      key: ai-agent-framework/database
      property: url
  - secretKey: username
    remoteRef:
      key: ai-agent-framework/database
      property: username
  - secretKey: password
    remoteRef:
      key: ai-agent-framework/database
      property: password
```

---

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: ai-agent-framework
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "/etc/prometheus/rules/*.yml"
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
    
    scrape_configs:
      - job_name: 'ai-agent-framework'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - ai-agent-framework
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
    
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
          - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
            action: keep
            regex: default;kubernetes;https
    
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
          - role: node
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)
          - target_label: __address__
            replacement: kubernetes.default.svc:443
          - source_labels: [__meta_kubernetes_node_name]
            regex: (.+)
            target_label: __metrics_path__
            replacement: /api/v1/nodes/${1}/proxy/metrics

  alert-rules.yml: |
    groups:
    - name: ai-agent-framework
      rules:
      - alert: HighErrorRate
        expr: rate(agent_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(agent_execution_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is crash looping"
      
      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "Cannot connect to database"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "AI Agent Framework - Production Dashboard",
    "tags": ["ai-agent-framework", "production"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(agent_executions_total[5m]))",
            "legendFormat": "Requests/sec"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(agent_errors_total[5m])) / sum(rate(agent_executions_total[5m])) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "yAxes": [
          {
            "label": "Error Rate %",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(agent_execution_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(agent_execution_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(agent_execution_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "99th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Response Time (s)",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Active Agents",
        "type": "stat",
        "targets": [
          {
            "expr": "count(up{job=\"agent-orchestrator\"} == 1)",
            "legendFormat": "Active Agents"
          }
        ],
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 16
        }
      },
      {
        "id": 5,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(container_memory_usage_bytes{pod=~\"agent-orchestrator-.*\"}) / 1024 / 1024 / 1024",
            "legendFormat": "Memory Usage (GB)"
          }
        ],
        "yAxes": [
          {
            "label": "Memory (GB)",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 6,
          "y": 16
        }
      },
      {
        "id": 6,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(rate(container_cpu_usage_seconds_total{pod=~\"agent-orchestrator-.*\"}[5m])) * 100",
            "legendFormat": "CPU Usage %"
          }
        ],
        "yAxes": [
          {
            "label": "CPU %",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 18,
          "y": 16
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

---

## Performance Optimization

### Database Optimization

```sql
-- PostgreSQL production optimizations
-- postgresql.conf optimizations

-- Memory settings
shared_buffers = 4GB                    # 25% of RAM
effective_cache_size = 12GB             # 75% of RAM
work_mem = 256MB                        # For complex queries
maintenance_work_mem = 1GB              # For maintenance operations

-- Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
max_wal_size = 4GB
min_wal_size = 1GB

-- Connection settings
max_connections = 200
shared_preload_libraries = 'pg_stat_statements'

-- Query optimization
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200          # For SSD storage

-- Logging
log_min_duration_statement = 1000       # Log slow queries
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

-- Enable query statistics
track_activities = on
track_counts = on
track_io_timing = on
track_functions = pl
```

```sql
-- Database indexes for performance
CREATE INDEX CONCURRENTLY idx_agents_type_status ON agents(type, status);
CREATE INDEX CONCURRENTLY idx_executions_agent_created ON executions(agent_id, created_at);
CREATE INDEX CONCURRENTLY idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX CONCURRENTLY idx_knowledge_docs_category ON knowledge_documents(category, classification);

-- Partitioning for large tables
CREATE TABLE audit_logs_2025 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE executions_2025 PARTITION OF executions  
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

### Redis Configuration

```redis
# redis.conf production settings

# Memory management
maxmemory 8gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Persistence
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes

# AOF
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Network
tcp-backlog 511
timeout 300
tcp-keepalive 300

# Security
requirepass ${REDIS_PASSWORD}
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""

# Performance
hz 10
dynamic-hz yes
```

### Application Performance Tuning

```yaml
# performance-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: performance-config
  namespace: ai-agent-framework
data:
  # Node.js performance settings
  NODE_OPTIONS: "--max-old-space-size=2048 --optimize-for-size"
  UV_THREADPOOL_SIZE: "16"
  
  # Application settings
  MAX_CONCURRENT_EXECUTIONS: "100"
  CONNECTION_POOL_SIZE: "20"
  CACHE_TTL: "300"
  REQUEST_TIMEOUT: "30000"
  
  # Database settings
  DB_POOL_MIN: "5"
  DB_POOL_MAX: "20"
  DB_POOL_IDLE_TIMEOUT: "30000"
  DB_POOL_ACQUIRE_TIMEOUT: "10000"
  
  # Redis settings
  REDIS_POOL_MIN: "5"
  REDIS_POOL_MAX: "10"
  REDIS_CONNECT_TIMEOUT: "5000"
  REDIS_COMMAND_TIMEOUT: "5000"
```

---

## Disaster Recovery

### Backup Strategy

```yaml
# backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: ai-agent-framework
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15-alpine
            command:
            - /bin/sh
            - -c
            - |
              pg_dump ${DATABASE_URL} \
                --clean \
                --if-exists \
                --create \
                --verbose \
                | gzip > /backup/backup-$(date +%Y%m%d-%H%M%S).sql.gz
              
              # Upload to S3
              aws s3 cp /backup/backup-$(date +%Y%m%d-%H%M%S).sql.gz \
                s3://ai-agent-framework-backups/postgres/
              
              # Cleanup local backup
              rm /backup/backup-$(date +%Y%m%d-%H%M%S).sql.gz
              
              # Cleanup old S3 backups (keep last 30 days)
              aws s3 ls s3://ai-agent-framework-backups/postgres/ \
                | while read -r line; do
                  createDate=$(echo $line | awk '{print $1" "$2}')
                  createDate=$(date -d"$createDate" +%s)
                  olderThan=$(date -d'30 days ago' +%s)
                  if [[ $createDate -lt $olderThan ]]; then
                    fileName=$(echo $line | awk '{print $4}')
                    if [[ $fileName != "" ]]; then
                      aws s3 rm s3://ai-agent-framework-backups/postgres/$fileName
                    fi
                  fi
                done
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-secret
                  key: url
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-access-key
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

### Multi-Region Setup

```yaml
# multi-region-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ai-agent-framework-us-west
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/company/ai-agent-framework-deploy
    targetRevision: HEAD
    path: manifests/us-west-2
  destination:
    server: https://us-west-2.eks.amazonaws.com
    namespace: ai-agent-framework
  syncPolicy:
    automated:
      prune: true
      selfHeal: true

---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ai-agent-framework-us-east
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/company/ai-agent-framework-deploy
    targetRevision: HEAD
    path: manifests/us-east-1
  destination:
    server: https://us-east-1.eks.amazonaws.com
    namespace: ai-agent-framework
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Database Replication

```yaml
# postgres-ha.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: ai-agent-framework
spec:
  instances: 3
  primaryUpdateStrategy: unsupervised
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "4GB"
      effective_cache_size: "12GB"
      wal_level: "replica"
      max_wal_senders: "10"
      max_replication_slots: "10"
      hot_standby: "on"
  
  bootstrap:
    initdb:
      database: ai_agent_framework
      owner: app_user
      secret:
        name: postgres-credentials
  
  storage:
    size: 1Ti
    storageClass: fast-ssd
  
  backup:
    retentionPolicy: "30d"
    barmanObjectStore:
      destinationPath: "s3://ai-agent-framework-backups/postgres"
      s3Credentials:
        accessKeyId:
          name: backup-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: backup-credentials
          key: SECRET_ACCESS_KEY
      wal:
        retention: "7d"
      data:
        retention: "30d"
  
  monitoring:
    enabled: true
    podMonitorSelector:
      matchLabels:
        app: postgres-cluster
```

---

## Scaling Strategies

### Vertical Pod Autoscaler

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: agent-orchestrator-vpa
  namespace: ai-agent-framework
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-orchestrator
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: agent-orchestrator
      maxAllowed:
        cpu: "2"
        memory: "4Gi"
      minAllowed:
        cpu: "100m"
        memory: "128Mi"
      controlledResources: ["cpu", "memory"]
```

### Cluster Autoscaler

```yaml
# cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/ai-agent-framework-cluster
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        env:
        - name: AWS_REGION
          value: us-west-2
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

```bash
# Debug memory usage
kubectl top pods -n ai-agent-framework --sort-by=memory

# Check memory limits
kubectl describe pod <pod-name> -n ai-agent-framework

# Get detailed memory metrics
kubectl exec -it <pod-name> -n ai-agent-framework -- cat /proc/meminfo

# Check for memory leaks
kubectl logs <pod-name> -n ai-agent-framework --previous
```

#### 2. Database Connection Issues

```bash
# Check database connectivity
kubectl exec -it postgres-0 -n ai-agent-framework -- psql -U postgres -c "SELECT version();"

# Check connection pool status
kubectl logs deployment/agent-orchestrator -n ai-agent-framework | grep "connection pool"

# Test database from application pod
kubectl exec -it <agent-pod> -n ai-agent-framework -- \
  curl -X GET http://localhost:3000/health/db
```

#### 3. Performance Degradation

```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -n ai-agent-framework

# Review slow queries
kubectl exec -it postgres-0 -n ai-agent-framework -- \
  psql -U postgres -c "SELECT query, calls, total_time, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Check cache hit rates
kubectl exec -it redis-0 -n ai-agent-framework -- \
  redis-cli info stats | grep keyspace
```

#### 4. Network Issues

```bash
# Test network connectivity
kubectl exec -it <pod-name> -n ai-agent-framework -- \
  nslookup postgres-service

# Check network policies
kubectl get networkpolicies -n ai-agent-framework

# Test service connectivity
kubectl exec -it <pod-name> -n ai-agent-framework -- \
  curl -v http://agent-orchestrator-service/health
```

### Debugging Commands

```bash
#!/bin/bash
# debug-cluster.sh

echo "=== Cluster Status ==="
kubectl get nodes -o wide

echo "=== AI Agent Framework Pods ==="
kubectl get pods -n ai-agent-framework -o wide

echo "=== Resource Usage ==="
kubectl top nodes
kubectl top pods -n ai-agent-framework

echo "=== Recent Events ==="
kubectl get events -n ai-agent-framework --sort-by='.lastTimestamp' | tail -20

echo "=== Service Status ==="
kubectl get services -n ai-agent-framework

echo "=== Ingress Status ==="
kubectl get ingress -n ai-agent-framework

echo "=== PVC Status ==="
kubectl get pvc -n ai-agent-framework

echo "=== HPA Status ==="
kubectl get hpa -n ai-agent-framework

echo "=== Network Policies ==="
kubectl get networkpolicies -n ai-agent-framework
```

### Health Check Endpoints

```javascript
// health-check.js
const http = require('http');
const { Pool } = require('pg');
const redis = require('redis');

class HealthChecker {
  constructor() {
    this.dbPool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 1,
      connectionTimeoutMillis: 5000,
    });
    
    this.redisClient = redis.createClient({
      url: process.env.REDIS_URL,
      socket: {
        connectTimeout: 5000,
      },
    });
  }

  async checkDatabase() {
    try {
      const result = await this.dbPool.query('SELECT 1');
      return { status: 'healthy', latency: Date.now() };
    } catch (error) {
      return { status: 'unhealthy', error: error.message };
    }
  }

  async checkRedis() {
    try {
      await this.redisClient.ping();
      return { status: 'healthy', latency: Date.now() };
    } catch (error) {
      return { status: 'unhealthy', error: error.message };
    }
  }

  async checkOverall() {
    const checks = await Promise.allSettled([
      this.checkDatabase(),
      this.checkRedis(),
    ]);

    const results = {
      database: checks[0].value || { status: 'error', error: checks[0].reason },
      redis: checks[1].value || { status: 'error', error: checks[1].reason },
    };

    const isHealthy = Object.values(results).every(check => check.status === 'healthy');
    
    return {
      status: isHealthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      checks: results,
    };
  }
}

// HTTP health check server
const healthChecker = new HealthChecker();

const server = http.createServer(async (req, res) => {
  if (req.url === '/health') {
    const health = await healthChecker.checkOverall();
    
    res.writeHead(health.status === 'healthy' ? 200 : 503, {
      'Content-Type': 'application/json',
    });
    res.end(JSON.stringify(health, null, 2));
  } else {
    res.writeHead(404);
    res.end('Not Found');
  }
});

server.listen(8080, () => {
  console.log('Health check server running on port 8080');
});
```

---

Esta guía de despliegue de producción proporciona patrones empresariales probados para implementar el AI Agent Framework a escala, siguiendo las mejores prácticas de organizaciones líderes en la industria como OpenAI, Google, Microsoft y Anthropic.

¿Te gustaría que continúe con los **Performance Benchmarks** o el **Security & Ethics Framework**?
