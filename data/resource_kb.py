"""
Resource Knowledge Base
────────────────────────
ChromaDB-free implementation:
  - Resources stored in a simple Python dict keyed by skill
  - Semantic similarity computed directly via sentence-transformers cosine similarity
  - No external vector DB, no protobuf conflicts, no ONNX download on cloud
"""
from __future__ import annotations
import os
import numpy as np

# Lazy-loaded sentence-transformer model
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# --- Curated Learning Resources (keyed by skill for O(1) lookup) ---
_RESOURCES: dict[str, list[dict]] = {
    "Python": [
        {"title": "Python Official Tutorial", "url": "https://docs.python.org/3/tutorial/", "type": "documentation", "hours": 8},
        {"title": "Real Python – Advanced Python", "url": "https://realpython.com/tutorials/advanced/", "type": "article", "hours": 10},
        {"title": "Build a CLI Tool with Click", "url": "https://realpython.com/python-click/", "type": "project", "hours": 5},
    ],
    "Django": [
        {"title": "Django Official Tutorial", "url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/", "type": "documentation", "hours": 6},
        {"title": "Django REST Framework Tutorial", "url": "https://www.django-rest-framework.org/tutorial/quickstart/", "type": "documentation", "hours": 8},
        {"title": "Build a Blog API – Project", "url": "https://learndjango.com/tutorials/django-rest-framework-tutorial-todo-api", "type": "project", "hours": 10},
    ],
    "FastAPI": [
        {"title": "FastAPI Official Tutorial", "url": "https://fastapi.tiangolo.com/tutorial/", "type": "documentation", "hours": 6},
        {"title": "FastAPI + PostgreSQL CRUD", "url": "https://testdriven.io/blog/fastapi-crud/", "type": "project", "hours": 8},
    ],
    "Docker": [
        {"title": "Docker Get Started", "url": "https://docs.docker.com/get-started/", "type": "documentation", "hours": 4},
        {"title": "Dockerize a Python App", "url": "https://docs.docker.com/language/python/", "type": "project", "hours": 3},
        {"title": "Docker Compose for Django + PostgreSQL", "url": "https://docs.docker.com/samples/django/", "type": "project", "hours": 4},
    ],
    "SQL": [
        {"title": "SQLZoo – Interactive SQL Tutorial", "url": "https://sqlzoo.net/wiki/SQL_Tutorial", "type": "course", "hours": 8},
        {"title": "Mode SQL Tutorial – Intermediate", "url": "https://mode.com/sql-tutorial/", "type": "course", "hours": 5},
        {"title": "Use The Index, Luke – Query Optimization", "url": "https://use-the-index-luke.com/", "type": "documentation", "hours": 6},
    ],
    "PostgreSQL": [
        {"title": "PostgreSQL Official Tutorial", "url": "https://www.postgresql.org/docs/current/tutorial.html", "type": "documentation", "hours": 5},
        {"title": "PostgreSQL Crash Course", "url": "https://www.youtube.com/watch?v=qw--VYLpxG4", "type": "video", "hours": 4},
    ],
    "REST APIs": [
        {"title": "REST API Design Best Practices", "url": "https://www.freecodecamp.org/news/rest-api-design-best-practices-build-a-rest-api/", "type": "article", "hours": 3},
        {"title": "HTTP Status Codes Reference – MDN", "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status", "type": "documentation", "hours": 1},
    ],
    "Machine Learning": [
        {"title": "fast.ai Practical Deep Learning", "url": "https://course.fast.ai/", "type": "course", "hours": 40},
        {"title": "Kaggle Learn – Intro to ML", "url": "https://www.kaggle.com/learn/intro-to-machine-learning", "type": "course", "hours": 5},
        {"title": "Hands-On ML – Aurélien Géron", "url": "https://github.com/ageron/handson-ml3", "type": "book", "hours": 60},
    ],
    "System Design": [
        {"title": "System Design Primer – GitHub", "url": "https://github.com/donnemartin/system-design-primer", "type": "documentation", "hours": 20},
        {"title": "Grokking System Design Interview", "url": "https://www.educative.io/courses/grokking-the-system-design-interview", "type": "course", "hours": 30},
    ],
    "Microservices": [
        {"title": "Microservices.io Patterns", "url": "https://microservices.io/patterns/", "type": "documentation", "hours": 8},
        {"title": "Build Microservices with FastAPI", "url": "https://testdriven.io/blog/fastapi-microservices/", "type": "project", "hours": 12},
    ],
    "React": [
        {"title": "React Official Docs – Learn React", "url": "https://react.dev/learn", "type": "documentation", "hours": 10},
        {"title": "Full Stack Open – React Module", "url": "https://fullstackopen.com/en/part1", "type": "course", "hours": 15},
    ],
    "AWS": [
        {"title": "AWS Free Tier – Hands-on Labs", "url": "https://aws.amazon.com/free/", "type": "project", "hours": 10},
        {"title": "AWS Cloud Practitioner Essentials", "url": "https://aws.amazon.com/training/digital/aws-cloud-practitioner-essentials/", "type": "course", "hours": 6},
    ],
    "Kubernetes": [
        {"title": "Kubernetes Official Tutorial", "url": "https://kubernetes.io/docs/tutorials/", "type": "documentation", "hours": 8},
        {"title": "Deploy a Django App on Kubernetes", "url": "https://testdriven.io/blog/django-kubernetes/", "type": "project", "hours": 10},
    ],
    "LangChain": [
        {"title": "LangChain Official Docs", "url": "https://python.langchain.com/docs/get_started/introduction", "type": "documentation", "hours": 8},
        {"title": "Build a RAG App – Project", "url": "https://python.langchain.com/docs/use_cases/question_answering/", "type": "project", "hours": 6},
    ],
    "CI/CD": [
        {"title": "GitHub Actions Official Docs", "url": "https://docs.github.com/en/actions", "type": "documentation", "hours": 5},
        {"title": "CI/CD for Django – Practical Guide", "url": "https://testdriven.io/blog/django-github-actions/", "type": "project", "hours": 4},
    ],
    "Redis": [
        {"title": "Redis Official Tutorial", "url": "https://redis.io/docs/manual/", "type": "documentation", "hours": 4},
        {"title": "Redis with Django – Caching", "url": "https://realpython.com/caching-in-django-with-redis/", "type": "project", "hours": 3},
    ],
}

# --- Gold standard answers for semantic similarity scoring ---
_GOLD_ANSWERS: dict[str, str] = {
    "Python": (
        "I use Python extensively for building scalable backend services. "
        "I leverage features like generators, context managers, dataclasses, and async/await. "
        "I write type-annotated code and use pytest for comprehensive testing."
    ),
    "Django": (
        "I build production Django applications with custom middleware, signals, and celery tasks. "
        "I optimise ORM queries using select_related, prefetch_related, and database indexes. "
        "I've implemented DRF APIs with JWT authentication and custom permission classes."
    ),
    "FastAPI": (
        "I use FastAPI with Pydantic models for strict input validation and automatic OpenAPI docs. "
        "I implement async endpoints with SQLAlchemy async sessions and Redis caching. "
        "I write dependency injection using FastAPI's Depends system for clean, testable code."
    ),
    "Docker": (
        "I containerise applications using multi-stage Dockerfiles to minimise image size. "
        "I use Docker Compose for local development with services like PostgreSQL and Redis. "
        "I've pushed images to ECR and deployed via ECS in production environments."
    ),
    "SQL": (
        "I write complex SQL queries with CTEs, window functions, and subqueries. "
        "I analyse query execution plans and add indexes to eliminate full table scans. "
        "I have experience with both OLTP and OLAP query patterns."
    ),
    "PostgreSQL": (
        "I use PostgreSQL-specific features like JSONB columns, full-text search, and advisory locks. "
        "I manage schema migrations carefully with rollback strategies. "
        "I've configured read replicas and connection pooling with PgBouncer for high-traffic systems."
    ),
    "Machine Learning": (
        "I build end-to-end ML pipelines: data cleaning, feature engineering, model training, and evaluation. "
        "I've worked with scikit-learn, XGBoost, and PyTorch. "
        "I track experiments with MLflow and deploy models as REST APIs."
    ),
    "System Design": (
        "I design systems by first clarifying requirements and estimating scale. "
        "I consider CAP theorem trade-offs, use caching strategically, and design for horizontal scalability. "
        "I document designs with sequence diagrams and capacity planning spreadsheets."
    ),
    "REST APIs": (
        "I design RESTful APIs following proper resource naming, HTTP verb semantics, and status codes. "
        "I implement versioning, rate limiting, and pagination. "
        "I document APIs with OpenAPI/Swagger specs and write integration tests."
    ),
    "Microservices": (
        "I've decomposed monoliths into microservices using domain-driven design principles. "
        "I implement inter-service communication with REST and async message queues (RabbitMQ/Kafka). "
        "I handle distributed tracing, circuit breakers, and idempotent operations."
    ),
}


def get_resources_for_skill(skill: str, top_k: int = 3) -> list[dict]:
    """Return top-k curated resources for the given skill."""
    resources = _RESOURCES.get(skill, [])
    if not resources:
        # Fuzzy fallback: try case-insensitive match
        skill_lower = skill.lower()
        for key, value in _RESOURCES.items():
            if skill_lower in key.lower() or key.lower() in skill_lower:
                resources = value
                break
    return resources[:top_k]


def get_semantic_similarity(answer: str, skill: str) -> float:
    """
    Compute cosine similarity between the candidate's answer
    and the gold-standard answer for the skill, using sentence-transformers.
    Returns a score in [0.0, 1.0].
    """
    gold = _GOLD_ANSWERS.get(skill)
    if not gold or not answer.strip():
        return 0.5  # neutral score if no gold answer exists

    try:
        model = _get_model()
        embeddings = model.encode([answer, gold], normalize_embeddings=True)
        # Cosine similarity = dot product of normalised vectors
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        # Clamp to [0, 1]
        return round(max(0.0, min(1.0, similarity)), 3)
    except Exception:
        return 0.5  # graceful fallback
