#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Unable to identify the operating system." >&2
  exit 1
fi

# shellcheck source=/dev/null
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This setup script supports Ubuntu only." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  gnupg \
  jq

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

architecture="$(dpkg --print-architecture)"
codename="${VERSION_CODENAME:?Ubuntu codename is unavailable}"
printf '%s\n' \
  "deb [arch=${architecture} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${codename} stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y --no-install-recommends \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

systemctl enable --now docker
docker version >/dev/null
docker compose version >/dev/null

login_user="${SUDO_USER:-}"
if [[ -n "${login_user}" && "${login_user}" != "root" ]]; then
  usermod -aG docker "${login_user}"
  echo "Docker installed. Sign out and back in before deploying as ${login_user}."
else
  echo "Docker Engine and Docker Compose installed successfully."
fi
