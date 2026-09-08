# WorkAssist AI Nginx configuration

Replace `WORKASSIST_DOMAIN` in both configuration files with the public DNS
name, such as `workassist.example.com`.

Before a TLS certificate exists, select the HTTP-only configuration and start
the stack:

```powershell
$env:NGINX_CONFIG_FILE = "./nginx/nginx.http.conf"
docker compose up -d
```

After placing the certificate chain at `nginx/certs/fullchain.pem` and private
key at `nginx/certs/privkey.pem`, switch to HTTPS:

```powershell
Remove-Item Env:NGINX_CONFIG_FILE -ErrorAction SilentlyContinue
docker compose up -d --force-recreate nginx
```

The default `nginx/nginx.conf` redirects HTTP to HTTPS. Only Nginx is published;
the `frontend:8501` and `backend:8000` services remain private to Docker.
