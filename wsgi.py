"""WSGI entrypoint for production servers (gunicorn, uwsgi)."""
from deepface_server import create_app

app = create_app()
