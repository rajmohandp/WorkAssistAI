"""Static deployment contracts that require no external provider connections."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"


def _service_block(service: str) -> str:
    """Return one top-level Compose service block without requiring PyYAML."""

    lines = COMPOSE_PATH.read_text(encoding="utf-8").splitlines()
    marker = f"  {service}:"
    start = lines.index(marker)
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith(" "):
            end = index
            break
        if line.startswith("  ") and not line.startswith("    "):
            end = index
            break
    return "\n".join(lines[start:end])


def test_only_nginx_publishes_public_ports():
    backend = _service_block("backend")
    frontend = _service_block("frontend")
    nginx = _service_block("nginx")

    assert "\n    ports:" not in backend
    assert "\n    ports:" not in frontend
    assert '      - "80:80"' in nginx
    assert '      - "443:443"' in nginx
    assert "3306:3306" not in COMPOSE_PATH.read_text(encoding="utf-8")
    assert '      - "8000"' in backend
    assert '      - "8501"' in frontend


def test_services_have_restart_and_health_contracts():
    for service in ("backend", "frontend", "nginx"):
        block = _service_block(service)
        assert "restart: unless-stopped" in block

    assert "healthcheck:" in _service_block("backend")
    assert "healthcheck:" in _service_block("nginx")
    assert "http://127.0.0.1:8000/health" in _service_block("backend")
    assert "http://127.0.0.1/nginx-health" in _service_block("nginx")
    frontend_dockerfile = (
        PROJECT_ROOT / "frontend" / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "HEALTHCHECK" in frontend_dockerfile
    assert "http://127.0.0.1:8501/_stcore/health" in frontend_dockerfile


def test_frontend_uses_private_backend_and_waits_for_health():
    frontend = _service_block("frontend")

    assert "BACKEND_URL: http://backend:8000" in frontend
    assert "backend:" in frontend
    assert "condition: service_healthy" in frontend


def test_docker_build_context_excludes_secrets_and_credentials():
    ignored = {
        line.strip()
        for line in (PROJECT_ROOT / ".dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".git", ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx"} <= ignored

    sensitive_names = {
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "AUTH_USERS_JSON",
    }
    for relative_path in ("backend/Dockerfile", "frontend/Dockerfile"):
        dockerfile = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "COPY . " not in dockerfile
        assert not sensitive_names.intersection(dockerfile.split())
        assert "ARG " not in dockerfile
        assert ".env" not in dockerfile


def test_deploy_script_verifies_health_and_preserves_rollback_images():
    script = (
        PROJECT_ROOT / "deployment" / "aws" / "deploy.sh"
    ).read_text(encoding="utf-8")

    assert "wait_for_health" in script
    assert "restore_previous_release" in script
    assert "rollback-${deployment_id}" in script
    assert 'git reset --hard "${previous_commit}"' in script
    assert "http://127.0.0.1:8000/ready" in script
    assert "required_environment_keys" in script
    assert "Remove static AWS credentials from .env" in script
    assert "Replace WORKASSIST_DOMAIN" in script
    assert "chmod 600 .env" in script
    assert "docker compose" in script
    forbidden = (
        "docker image prune",
        "docker system prune",
        "docker volume rm",
        "docker compose down -v",
        "docker compose down --volumes",
    )
    assert all(command not in script for command in forbidden)


def test_ec2_setup_does_not_install_a_database():
    script = (
        PROJECT_ROOT / "deployment" / "aws" / "setup-ec2.sh"
    ).read_text(encoding="utf-8").casefold()

    assert "mysql-server" not in script
    assert "mariadb-server" not in script
    assert "postgresql" not in script
