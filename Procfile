# ─────────────────────────────────────────────────────────────────
# Procfile — Tells Railway (and Heroku/Render) how to start the app
# ─────────────────────────────────────────────────────────────────
web: gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload
