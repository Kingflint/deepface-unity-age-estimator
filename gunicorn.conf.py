"""Gunicorn configuration for production deployments."""
import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get('GUNICORN_WORKERS', max(1, multiprocessing.cpu_count() // 2)))
threads = int(os.environ.get('GUNICORN_THREADS', 2))
worker_class = 'gthread'
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 60))
graceful_timeout = 30
keepalive = 5
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('LOG_LEVEL', 'info').lower()
