# TLS certificates

Place the public TLS certificate chain at `fullchain.pem` and its private key at
`privkey.pem` before starting the production Compose stack. Certificate and key
files are excluded from Git and the Docker build context.
