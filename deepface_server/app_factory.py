"""Application factory."""
from flask import Flask, jsonify

from .config import load_settings
from .errors import DeepFaceServerError
from .logging_config import configure_logging
from .services.deepface_service import DeepFaceService
from .services.image_service import ImageService


def create_app(settings=None):
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    app = Flask(__name__)
    app.config["SETTINGS"] = settings

    app.extensions["image_service"] = ImageService()
    app.extensions["deepface_service"] = DeepFaceService()

    from .blueprints.analyze import analyze_bp
    from .blueprints.health import health_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(analyze_bp)

    @app.errorhandler(DeepFaceServerError)
    def _on_known(err):
        return jsonify(err.to_dict()), err.status_code

    return app