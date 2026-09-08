#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd -- "${script_directory}/../.." && pwd)"
backup_directory="${BACKUP_DIRECTORY:-/var/backups/workassist}"
passphrase_file="${BACKUP_PASSPHRASE_FILE:-}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_file="${backup_directory}/workassist-config-${timestamp}.tar.gz.gpg"

if [[ -z "${passphrase_file}" || ! -r "${passphrase_file}" ]]; then
  echo "Set BACKUP_PASSPHRASE_FILE to a readable root-owned passphrase file." >&2
  exit 1
fi
passphrase_mode="$(stat --format='%a' "${passphrase_file}")"
if (( (8#${passphrase_mode} & 077) != 0 )); then
  echo "The backup passphrase file must not be accessible by group or others." >&2
  exit 1
fi
if ! command -v gpg >/dev/null 2>&1; then
  echo "gpg is required to encrypt configuration backups." >&2
  exit 1
fi

required_paths=(.env certs/global-bundle.pem)
optional_paths=(nginx/certs/fullchain.pem nginx/certs/privkey.pem)
backup_paths=()
for path in "${required_paths[@]}"; do
  if [[ ! -f "${project_directory}/${path}" ]]; then
    echo "Required configuration file is missing: ${path}" >&2
    exit 1
  fi
  backup_paths+=("${path}")
done
for path in "${optional_paths[@]}"; do
  if [[ -f "${project_directory}/${path}" ]]; then
    backup_paths+=("${path}")
  fi
done

umask 077
install -d -m 0700 "${backup_directory}"
tar -C "${project_directory}" -czf - "${backup_paths[@]}" \
  | gpg --batch --yes --symmetric --cipher-algo AES256 \
      --passphrase-file "${passphrase_file}" --output "${backup_file}"

echo "Encrypted configuration backup created at ${backup_file}."
