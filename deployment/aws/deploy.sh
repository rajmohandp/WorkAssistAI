#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd -- "${script_directory}/../.." && pwd)"
compose_file="${project_directory}/docker-compose.yml"
deploy_remote="${DEPLOY_REMOTE:-origin}"
deploy_branch="${DEPLOY_BRANCH:-main}"
health_timeout_seconds="${HEALTH_TIMEOUT_SECONDS:-180}"
deployment_id="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! "${health_timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "HEALTH_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 1
fi

cd "${project_directory}"

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker Engine and the Docker Compose plugin are required." >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "Missing .env. Create it from .env.example before deployment." >&2
  exit 1
fi

env_value() {
  local requested_key="$1"
  awk -v requested_key="${requested_key}" '
    /^[[:space:]]*#/ { next }
    {
      separator = index($0, "=")
      if (!separator) next
      key = substr($0, 1, separator - 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key == requested_key) {
        value = substr($0, separator + 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
      }
    }
  ' .env
}

required_environment_keys=(
  APP_ENV
  BACKEND_URL
  OPENAI_API_KEY
  PINECONE_API_KEY
  PINECONE_INDEX_NAME
  AWS_REGION
  S3_BUCKET_NAME
  DATABASE_URL
  DB_SSL_CA_PATH
  JWT_SECRET_KEY
  AUTH_USERS_JSON
  ALLOWED_ORIGINS
)
missing_environment_keys=()
for key in "${required_environment_keys[@]}"; do
  if [[ -z "$(env_value "${key}")" ]]; then
    missing_environment_keys+=("${key}")
  fi
done
if (( ${#missing_environment_keys[@]} > 0 )); then
  printf 'Missing required .env variables: %s\n' \
    "${missing_environment_keys[*]}" >&2
  exit 1
fi
if [[ "$(env_value APP_ENV)" != "production" ]]; then
  echo "APP_ENV must be production on the EC2 deployment host." >&2
  exit 1
fi
if [[ -n "$(env_value AWS_ACCESS_KEY_ID)" || -n "$(env_value AWS_SECRET_ACCESS_KEY)" ]]; then
  echo "Remove static AWS credentials from .env; use the EC2 IAM role." >&2
  exit 1
fi
if [[ -n "$(env_value AWS_PROFILE)" ]]; then
  echo "AWS_PROFILE must be empty on EC2 so the instance role is used." >&2
  exit 1
fi

env_permissions="$(stat --format='%a' .env)"
if (( (8#${env_permissions} & 077) != 0 )); then
  echo ".env must not be accessible by group or other users; run chmod 600 .env." >&2
  exit 1
fi
if [[ ! -f certs/global-bundle.pem ]]; then
  echo "Missing certs/global-bundle.pem for verified RDS TLS." >&2
  exit 1
fi

nginx_config="$(env_value NGINX_CONFIG_FILE)"
nginx_config="${nginx_config:-./nginx/nginx.conf}"
if [[ "${nginx_config}" = /* ]]; then
  resolved_nginx_config="${nginx_config}"
else
  resolved_nginx_config="${project_directory}/${nginx_config#./}"
fi
if [[ ! -f "${resolved_nginx_config}" ]]; then
  echo "The selected Nginx configuration file does not exist." >&2
  exit 1
fi
if grep --quiet 'WORKASSIST_DOMAIN' "${resolved_nginx_config}"; then
  echo "Replace WORKASSIST_DOMAIN in the selected Nginx configuration." >&2
  exit 1
fi
if grep --quiet 'listen 443 ssl' "${resolved_nginx_config}"; then
  if [[ ! -f nginx/certs/fullchain.pem || ! -f nginx/certs/privkey.pem ]]; then
    echo "The HTTPS Nginx configuration requires its certificate files." >&2
    exit 1
  fi
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Deployment stopped because the EC2 worktree has uncommitted changes." >&2
  exit 1
fi
previous_commit="$(git rev-parse --verify HEAD)"

declare -A previous_images=()
declare -A rollback_tags=()
services=(backend frontend nginx)

for service in "${services[@]}"; do
  container_id="$(docker compose -f "${compose_file}" ps -q "${service}" 2>/dev/null || true)"
  if [[ -n "${container_id}" ]]; then
    image_id="$(docker inspect --format '{{.Image}}' "${container_id}")"
    rollback_tag="workassist-${service}:rollback-${deployment_id}"
    docker image tag "${image_id}" "${rollback_tag}"
    previous_images["${service}"]="${image_id}"
    rollback_tags["${service}"]="${rollback_tag}"
  fi
done

rollback_file="$(mktemp)"
cleanup() {
  rm -f -- "${rollback_file}"
}
trap cleanup EXIT

write_rollback_override() {
  {
    echo "services:"
    for service in "${services[@]}"; do
      if [[ -n "${rollback_tags[${service}]:-}" ]]; then
        printf '  %s:\n    image: %s\n' "${service}" "${rollback_tags[${service}]}"
      fi
    done
  } > "${rollback_file}"
  chmod 600 "${rollback_file}"
}

wait_for_health() {
  local service="$1"
  local deadline=$((SECONDS + health_timeout_seconds))
  local container_id health_status running_status

  while (( SECONDS < deadline )); do
    container_id="$(docker compose -f "${compose_file}" ps -q "${service}" 2>/dev/null || true)"
    if [[ -n "${container_id}" ]]; then
      running_status="$(docker inspect --format '{{.State.Status}}' "${container_id}")"
      health_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
      if [[ "${running_status}" == "running" && "${health_status}" == "healthy" ]]; then
        return 0
      fi
      if [[ "${running_status}" == "exited" || "${health_status}" == "unhealthy" ]]; then
        return 1
      fi
    fi
    sleep 2
  done
  return 1
}

restore_previous_release() {
  echo "New release failed verification; restoring the previous running images." >&2
  if [[ "${#previous_images[@]}" -ne "${#services[@]}" ]]; then
    echo "A complete previous release was not available for automatic rollback." >&2
    docker compose -f "${compose_file}" ps
    return 1
  fi

  # The clean-worktree precondition makes this safe and ensures the prior
  # images also use their matching Compose and bind-mounted Nginx files.
  git reset --hard "${previous_commit}" >/dev/null
  write_rollback_override
  docker compose \
    -f "${compose_file}" \
    -f "${rollback_file}" \
    up -d --no-build

  for service in "${services[@]}"; do
    if ! wait_for_health "${service}"; then
      echo "Rollback container ${service} did not become healthy." >&2
      docker compose -f "${compose_file}" ps
      return 1
    fi
  done
  echo "Previous release restored. Investigate the failed release before retrying." >&2
}

echo "Pulling ${deploy_remote}/${deploy_branch}."
git pull --ff-only "${deploy_remote}" "${deploy_branch}"

# Quiet validation expands the Compose model without rendering environment values.
docker compose -f "${compose_file}" config --quiet

echo "Building updated service images."
docker compose -f "${compose_file}" build --pull backend frontend

echo "Validating backend configuration without contacting external services."
docker compose -f "${compose_file}" run --rm --no-deps --entrypoint python backend \
  -c "from src.config import get_environment_settings; get_environment_settings().validate_backend_startup(); print('Configuration validation passed.')"

echo "Starting the new release."
if ! docker compose -f "${compose_file}" up -d; then
  restore_previous_release
  exit 1
fi

for service in "${services[@]}"; do
  if ! wait_for_health "${service}"; then
    echo "Service ${service} failed its container health check." >&2
    restore_previous_release
    exit 1
  fi
done

echo "Validating database readiness."
if ! docker compose -f "${compose_file}" exec -T backend python -c \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=10).read()"; then
  restore_previous_release
  exit 1
fi

echo "Deployment completed successfully."
docker compose -f "${compose_file}" ps
