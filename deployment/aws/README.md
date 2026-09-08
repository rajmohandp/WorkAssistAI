# WorkAssist AI deployment on AWS EC2

This guide deploys WorkAssist AI in `us-east-2` (Ohio) using Ubuntu LTS,
one `t3.small` EC2 instance, a 25 GiB gp3 root volume, Docker Compose, Nginx,
Amazon RDS MySQL, Amazon S3, Pinecone, and OpenAI. Route 53 is optional.

Commands are explicitly labeled **Local workstation** or **EC2**. Replace every
uppercase placeholder; never commit `.env`, private keys, database passwords,
API keys, account IDs, or backup passphrases.

## 0. Cost and architecture notes

New eligible AWS customers can receive up to USD 200 in credits, and the Free
account plan currently lasts up to six months or until credits are depleted.
Eligibility and included services vary. A `t3.small`, 25 GiB gp3 volume, public
IPv4 address, RDS instance, Route 53 hosted zone, and S3 usage can consume
credits or incur charges. Check Billing, create a small AWS Budget, and enable
billing alerts before deployment. See the official
[AWS Free Tier guide](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier.html).

```text
Internet -> EC2:80/443 -> Nginx -> Streamlit:8501 -> FastAPI:8000
EC2 security group -> RDS security group:3306 -> private RDS MySQL
EC2 IAM role -> S3 bucket in us-east-2
```

Ports 8000 and 8501 exist only on the private Docker network. Port 3306 is
never opened to the internet.

## 1. Prepare AWS resources

In the AWS console, select **US East (Ohio) `us-east-2`** before creating EC2,
networking, or RDS connectivity.

Use the existing RDS MySQL database containing the WorkAssist AI schema and PTO
data. EC2 and RDS should be in the same VPC and Region. Set RDS **Public access**
to **No**. This deployment does not create, migrate, reset, or seed a database.

### 1.1 Create and attach the S3 IAM role

1. Open IAM -> Policies -> Create policy -> JSON.
2. Copy [`ec2-s3-policy.json`](./ec2-s3-policy.json), replacing both occurrences
   of `WORKASSIST_BUCKET` with the exact existing bucket name.
3. Name the policy `WorkAssistS3ReadOnly`.
4. Open IAM -> Roles -> Create role.
5. Choose **AWS service**, use case **EC2**, and attach that policy.
6. Name the role `WorkAssistEC2Role`.

The application needs only `s3:ListBucket` and `s3:GetObject`. Do not add S3
write or delete actions. If objects use a customer-managed KMS key, separately
grant `kms:Decrypt` on that one key. IAM roles supply temporary credentials, so
no static AWS keys belong in `.env`. See AWS's
[EC2 IAM-role guidance](https://docs.aws.amazon.com/autoscaling/ec2/userguide/us-iam-role.html).

Attach the role during EC2 launch in step 1.4. If the instance already exists,
select it -> Actions -> Security -> Modify IAM role, select
`WorkAssistEC2Role`, and choose Update IAM role. AWS also documents this
[existing-instance procedure](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/attach-iam-role.html).

### 1.2 Create the EC2 security group

Create `workassist-ec2-sg` in the application VPC with these inbound rules:

| Type | Port | Source | Purpose |
|---|---:|---|---|
| SSH | 22 | `ADMIN_PUBLIC_IP/32` | Administrator access only |
| HTTP | 80 | `0.0.0.0/0` | Web and Let's Encrypt validation |
| HTTPS | 443 | `0.0.0.0/0` | Public application traffic |

If IPv6 is enabled, add `::/0` only for ports 80 and 443. Never use an
anywhere rule for SSH. AWS likewise recommends restricting SSH to a specific
address; see its
[EC2 security-group guidance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-security-group.html).

Do **not** create inbound rules for 8000, 8501, or 3306. Leave normal outbound
access so the application can reach RDS, S3, Pinecone, OpenAI, package
repositories, and Let's Encrypt.

### 1.3 Restrict the RDS security group

On the security group attached to RDS, create this application rule:

| Type | Protocol | Port | Source |
|---|---|---:|---|
| MySQL/Aurora | TCP | 3306 | `workassist-ec2-sg` security-group ID |

The source must be the EC2 security group, never `0.0.0.0/0`. Remove public or
administrator-IP MySQL rules when no longer needed. AWS documents this pattern
in its [EC2-to-RDS tutorial](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/tutorial-connect-ec2-instance-to-rds-database.html).

### 1.4 Launch the EC2 instance

1. Open EC2 -> Instances -> Launch instances.
2. Name it `workassist-production`.
3. Select the latest Canonical **Ubuntu Server LTS** AMI for 64-bit x86.
4. Select **t3.small**.
5. Select or create an SSH key pair and protect the downloaded private key.
6. Choose the RDS VPC and a subnet with outbound internet access.
7. Enable a public IPv4 address for this single-instance deployment.
8. Select only `workassist-ec2-sg`.
9. Configure an encrypted **25 GiB gp3** root EBS volume.
10. Under Advanced details, select `WorkAssistEC2Role` and require IMDSv2.
11. Launch the instance.

For Docker containers to obtain the EC2 role through IMDSv2, set the instance
metadata **HTTP PUT response hop limit** to `2`: Actions -> Instance settings ->
Modify instance metadata options.

Allocate an Elastic IP in `us-east-2` and associate it before configuring DNS.
This provides a stable address, but public IPv4 addresses can consume credits
or incur charges.

## 2. Connect and install Docker

**Local workstation (macOS/Linux):**

```bash
chmod 400 /path/to/WORKASSIST_EC2_KEY.pem
ssh -i /path/to/WORKASSIST_EC2_KEY.pem ubuntu@EC2_PUBLIC_IP
```

Windows PowerShell uses the same `ssh` command; ensure the key is readable only
by your Windows account.

**EC2:**

```bash
git clone https://github.com/OWNER/WorkAssistAI.git
cd WorkAssistAI
chmod +x deployment/aws/setup-ec2.sh deployment/aws/deploy.sh
sudo deployment/aws/setup-ec2.sh
exit
```

The setup script installs Docker Engine, Compose, and utilities, but no
database. Reconnect so Docker group membership applies.

**Local workstation:**

```bash
ssh -i /path/to/WORKASSIST_EC2_KEY.pem ubuntu@EC2_PUBLIC_IP
```

**EC2:**

```bash
docker version
docker compose version
systemctl is-enabled docker
```

## 3. Create the production environment

**EC2:**

```bash
cd ~/WorkAssistAI
cp .env.example .env
chmod 600 .env
mkdir -p certs nginx/certs
chmod 700 certs nginx/certs
curl -fsSL https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
  -o certs/global-bundle.pem
chmod 644 certs/global-bundle.pem
nano .env
```

Use this placeholder-only shape:

```dotenv
APP_ENV=production
BACKEND_URL=http://backend:8000
ALLOWED_ORIGINS=http://WORKASSIST_DOMAIN,https://WORKASSIST_DOMAIN
LOG_LEVEL=INFO

OPENAI_API_KEY=REPLACE_WITH_OPENAI_KEY
PINECONE_API_KEY=REPLACE_WITH_PINECONE_KEY
PINECONE_INDEX_NAME=REPLACE_WITH_INDEX_NAME
PINECONE_CLOUD=aws
PINECONE_REGION=REPLACE_WITH_PINECONE_INDEX_REGION

AWS_REGION=us-east-2
S3_BUCKET_NAME=REPLACE_WITH_BUCKET_NAME
AWS_PROFILE=

DATABASE_URL=mysql+pymysql://REPLACE_USER:REPLACE_URL_ENCODED_PASSWORD@REPLACE_RDS_ENDPOINT:3306/REPLACE_DATABASE
DB_SSL_CA_PATH=/run/secrets/aws-rds-global-bundle.pem

JWT_SECRET_KEY=REPLACE_WITH_RANDOM_VALUE_AT_LEAST_32_CHARACTERS
AUTH_USERS_JSON=REPLACE_WITH_VALID_USER_RECORDS_JSON
AUTH_TOKEN_TTL_SECONDS=1800

CHAT_PROVIDER=openai
OPENAI_CHAT_MODEL=REPLACE_WITH_SUPPORTED_CHAT_MODEL
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1024
PINECONE_NAMESPACE=
PINECONE_BATCH_SIZE=100
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_TOP_K=5
RETRIEVAL_MIN_SCORE=0.3
RAG_EVALUATION_ENABLED=false

NGINX_CONFIG_FILE=./nginx/nginx.http.conf
```

URL-encode reserved database-password characters. Do not add AWS access-key
variables. Confirm file protection without printing its contents:

```bash
stat -c '%a %n' .env
```

The expected mode is `600`.

## 4. Configure HTTP and start Docker Compose

Choose the final host, such as `workassist.example.com`.

**EC2:** replace the placeholder in both Nginx configurations:

```bash
sed -i 's/WORKASSIST_DOMAIN/workassist.example.com/g' nginx/nginx.conf nginx/nginx.http.conf
grep -n 'server_name' nginx/nginx.conf nginx/nginx.http.conf
deployment/aws/deploy.sh
```

Use the actual domain. The deployment script pulls with `--ff-only`, validates
configuration, builds images, starts Compose, waits for health checks, checks
database readiness, and displays status. Compose publishes only Nginx ports 80
and 443. `restart: unless-stopped` plus the enabled Docker service restarts the
application after reboot unless an administrator explicitly stopped it.

## 5. Test health and Streamlit

**EC2:** test FastAPI internally because port 8000 is private:

```bash
docker compose exec -T backend python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read().decode())"
docker compose exec -T backend python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=10).read().decode())"
docker compose ps
```

`/health` checks the API process. `/ready` separately checks RDS.

**Local workstation:**

```bash
curl -I http://EC2_PUBLIC_IP
```

Open `http://EC2_PUBLIC_IP` and verify Streamlit login, RBAC, chat, PTO, and
escalation. The browser must never connect directly to ports 8000 or 8501.

## 6. Configure optional Route 53 DNS

If Route 53 hosts the domain:

1. Open Route 53 -> Hosted zones -> the desired hosted zone.
2. Choose Create record.
3. Enter the subdomain, such as `workassist`.
4. Choose type **A**, value `EC2_ELASTIC_IP`, routing policy **Simple**, TTL 300.
5. Save and wait for propagation.

For external DNS, create the equivalent A record there. AWS documents routing
a hostname to an [EC2 address](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-to-ec2-instance.html).

**Local workstation:**

```bash
nslookup workassist.example.com
curl -I http://workassist.example.com
```

Do not request TLS until DNS resolves and port 80 reaches temporary Nginx.

## 7. Install Let's Encrypt HTTPS

The Nginx container owns port 80, so stop only Nginx briefly for Certbot's
standalone HTTP-01 challenge.

**EC2:** replace the example domain and administrator email:

```bash
cd ~/WorkAssistAI
sudo apt-get update
sudo apt-get install -y certbot
docker compose stop nginx
sudo certbot certonly --standalone \
  --domain workassist.example.com \
  --email ADMIN_EMAIL_ADDRESS \
  --agree-tos \
  --no-eff-email
sudo install -m 0644 \
  /etc/letsencrypt/live/workassist.example.com/fullchain.pem \
  nginx/certs/fullchain.pem
sudo install -m 0600 \
  /etc/letsencrypt/live/workassist.example.com/privkey.pem \
  nginx/certs/privkey.pem
sudo chown -R ubuntu:ubuntu nginx/certs
sed -i 's|^NGINX_CONFIG_FILE=.*|NGINX_CONFIG_FILE=./nginx/nginx.conf|' .env
docker compose up -d nginx
docker compose ps
```

If issuance fails, run `docker compose up -d nginx`, fix DNS or firewall rules,
and retry.

**Local workstation:**

```bash
curl -I http://workassist.example.com
curl -I https://workassist.example.com
```

Open `https://workassist.example.com` and repeat the functional test. Then set
only the HTTPS origin in `.env`:

```dotenv
ALLOWED_ORIGINS=https://workassist.example.com
```

**EC2:** apply that change:

```bash
docker compose up -d --force-recreate backend
```

### Certificate renewal

Because Nginx runs in Docker, copy renewed files into its mounted folder with a
Certbot deploy hook.

**EC2:** replace the domain and project path if needed:

```bash
sudo tee /etc/letsencrypt/renewal-hooks/deploy/workassist-nginx.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
project=/home/ubuntu/WorkAssistAI
domain=workassist.example.com
install -m 0644 "/etc/letsencrypt/live/${domain}/fullchain.pem" "${project}/nginx/certs/fullchain.pem"
install -m 0600 "/etc/letsencrypt/live/${domain}/privkey.pem" "${project}/nginx/certs/privkey.pem"
chown -R ubuntu:ubuntu "${project}/nginx/certs"
docker compose --project-directory "${project}" -f "${project}/docker-compose.yml" restart nginx
EOF
sudo chmod 750 /etc/letsencrypt/renewal-hooks/deploy/workassist-nginx.sh
sudo certbot renew --dry-run
systemctl list-timers | grep certbot
```

## 8. Operate and update the deployment

### View status and logs

**EC2:**

```bash
cd ~/WorkAssistAI
docker compose ps
docker compose logs --tail=100 backend frontend nginx
docker compose logs --follow --tail=100 backend
```

Avoid unredacted `docker compose config` in shared logs because rendered output
can contain environment values.

### Deploy later updates

**Local workstation:**

```bash
git add PATHS_TO_CHANGED_FILES
git commit -m "Describe the WorkAssist AI update"
git push origin main
```

**EC2:**

```bash
cd ~/WorkAssistAI
deployment/aws/deploy.sh
```

Optional branch and health timeout:

```bash
DEPLOY_BRANCH=release HEALTH_TIMEOUT_SECONDS=300 deployment/aws/deploy.sh
```

### Roll back a failed update

Before replacement, `deploy.sh` records the current Git commit and tags the
complete running image set. If startup, any health check, or `/ready` fails, it
restores that commit and those prior images, then exits unsuccessfully. The
clean-worktree preflight prevents operator edits from being overwritten. It
never deletes volumes or prunes images.

**EC2:** confirm rollback and inspect the failure:

```bash
docker compose ps
docker image ls --filter 'reference=workassist-*:rollback-*'
docker compose logs --tail=200 backend frontend nginx
```

Do not deploy again until the cause is fixed. A first-ever deployment has no
prior release to restore; failed containers remain visible for diagnosis.

### Verify restart after reboot

**EC2:**

```bash
sudo reboot
```

**Local workstation:** reconnect after EC2 returns:

```bash
ssh -i /path/to/WORKASSIST_EC2_KEY.pem ubuntu@EC2_PUBLIC_IP
```

**EC2:**

```bash
cd ~/WorkAssistAI
docker compose ps
```

### Stop resources to conserve credits

Stopping containers alone does not stop EC2, RDS, EBS, public IPv4, Route 53,
or S3 charges.

**EC2:** stop containers without removing volumes:

```bash
cd ~/WorkAssistAI
docker compose stop
```

Then stop EC2 and, if an outage is acceptable, RDS in the AWS console. RDS
automatically restarts after its service-defined maximum stop period, so monitor
it. EBS and RDS storage, snapshots, S3, Route 53, and public IPv4 resources can
continue accruing charges. Release an Elastic IP only if DNS can be changed
before the next start.

To resume, start RDS first and wait for **Available**, then start EC2. Containers
manually stopped under `unless-stopped` may require:

```bash
cd ~/WorkAssistAI
docker compose start
```

Never use `docker compose down --volumes` or automated image pruning here.
