"""Application factory.

Exposes :func:`create_app` which wires up configuration, logging,
middleware, services and route blueprints. Tests use this instead of
importing a module-level Flask instance.
"""
from __future__ import annotations

from typing import Optional

from flask import Flask, jsonify

from . import __version__
from .config import Settings, load_settings
from .errors import DeepFaceServerError
from .logging_config import configure_logging, get_logger
from .middleware import register_middleware
from .services.cache import LRUCache
from .services.deepface_service import DeepFaceService
from .services.image_service import ImageService
from .services.metrics import MetricsCollector


def create_app(settings: Optional[Settings] = None) -> Flask:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info("starting deepface server v%s", __version__)

    app = Flask(__name__)
    app.config.update(
        SETTINGS=settings,
        VERSION=__version__,
        JSON_SORT_KEYS=False,
    )

    # Services
    cache = LRUCache(max_entries=settings.cache_max_entries) if settings.enable_cache else None
    image_service = ImageService(
        max_bytes=settings.max_image_bytes,
        max_dimension=settings.max_image_dimension,
    )
    deepface_service = DeepFaceService(
        actions=settings.deepface_actions,
        enforce_detection=settings.enforce_detection,
        detector_backend=settings.detector_backend,
    )
    metrics = MetricsCollector()

    app.extensions["image_service"] = image_service
    app.extensions["deepface_service"] = deepface_service
    app.extensions["cache"] = cache
    app.extensions["metrics"] = metrics

    register_middleware(app, settings)

    # Blueprints
    from .blueprints.analyze import analyze_bp
    from .blueprints.batch import batch_bp
    from .blueprints.health import health_bp
    from .blueprints.metrics import metrics_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(analyze_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(metrics_bp)

    register_error_handlers(app)
    return app


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(DeepFaceServerError)
    def _on_known(err: DeepFaceServerError):
        return jsonify(err.to_dict()), err.status_code

    @app.errorhandler(404)
    def _on_404(_):
        return jsonify({"error": "Not Found", "code": "not_found"}), 404

    @app.errorhandler(405)
    def _on_405(_):
        return jsonify({"error": "Method Not Allowed", "code": "method_not_allowed"}), 405

    @app.errorhandler(Exception)
    def _on_unexpected(err: Exception):
        get_logger(__name__).exception("unhandled error")
        return jsonify({"error": str(err), "code": "internal_error"}), 500
