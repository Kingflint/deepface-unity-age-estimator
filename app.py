"""Top-level entrypoint."""
from deepface_server import create_app
from deepface_server.config import load_settings

settings = load_settings()
app = create_app(settings)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.port, debug=settings.debug)