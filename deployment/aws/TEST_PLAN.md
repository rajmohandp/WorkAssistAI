# Deployment-focused test plan

All default automated tests run locally with provider boundaries mocked. They
must not use production AWS, RDS, Pinecone, or OpenAI credentials. Tests that
exercise real containers or AWS networking belong in an isolated staging
environment and are not part of the default suite.

Run the complete safe suite from the repository root:

```bash
python -m pytest -q
```

## Coverage matrix

| # | Deployment behavior | Automated evidence | Type |
|---:|---|---|---|
| 1 | Backend `/health` response | `tests/test_api.py::test_health_endpoint` and dependency-isolation test | Mocked API |
| 2 | Database readiness success and safe failure | `tests/test_api.py::test_readiness_endpoint_*`, `tests/test_database.py` | Mocked API/unit |
| 3 | Required configuration validation | `tests/test_environment_settings.py` | Unit |
| 4 | Streamlit-to-FastAPI URL, payload, timeout, and unavailable behavior | `tests/test_frontend_client.py`, deployment private-backend contract | Mocked HTTP/static |
| 5 | Authentication | `tests/test_security.py`, `tests/test_auth.py` | API/unit |
| 6 | Admin and employee RBAC | `tests/test_security.py`, `tests/test_agent_routing.py`, `tests/test_handoff.py` | API/unit |
| 7 | PTO requests use MySQL and not RAG | `tests/test_agent_routing.py::test_current_pto_calls_database_and_not_pinecone`, repository expression tests | Mocked routing/unit |
| 8 | Policy questions use RAG/Pinecone and not MySQL | policy cases in `tests/test_agent_routing.py` and `tests/test_handoff.py` | Mocked routing |
| 9 | S3 synchronization is admin-only and provider-safe | sync cases in `tests/test_security.py`, `tests/test_api.py`, and `tests/test_s3_loader.py` | Mocked API/unit |
| 10 | Human escalation and admin-only queue | `tests/test_handoff.py` | Mocked graph/API |
| 11 | Production CORS rejects wildcard origins | `tests/test_environment_settings.py::test_production_rejects_wildcard_cors` | Unit |
| 12 | Secret files and values are excluded from image build inputs | `tests/test_deployment_contract.py::test_docker_build_context_excludes_secrets_and_credentials` | Static contract |
| 13 | Ports 8000, 8501, and 3306 are not published | `tests/test_deployment_contract.py::test_only_nginx_publishes_public_ports` | Static Compose contract |
| 14 | Restart, health ordering, and rollback are configured | restart/health and deploy rollback tests in `tests/test_deployment_contract.py` | Static operational contract |
| 15 | Logs redact secrets and employee data | `tests/test_logging_config.py` and RAG lifecycle logging test | Unit/API |

## Optional staging-only checks

Use a staging `.env`, staging RDS database, staging S3 bucket, and separate
Pinecone namespace. Never point these checks at production data.

1. Build images and run `docker compose up -d` on a staging host.
2. Confirm `docker compose ps` reports all services healthy.
3. Inspect each built image filesystem and history for an injected, disposable
   sentinel value; fail if it appears. Do not inject real credentials.
4. Confirm only host ports 80 and 443 appear in `docker compose ps` and in the
   host firewall/socket listing.
5. Restart Docker, then confirm all services return healthy and `/ready` passes.
6. Deploy an intentionally unhealthy disposable image and confirm `deploy.sh`
   restores the previously healthy staging images.
7. Run login, employee PTO, admin sync authorization, policy RAG, and escalation
   smoke tests using synthetic staging identities and documents.

The staging namespace and synthetic records should be deleted through the
provider's normal reviewed cleanup process, never from the default test suite.

