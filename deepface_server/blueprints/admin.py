"""Admin endpoints for cache and registry inspection."""
from flask import Blueprint, current_app, jsonify

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/cache", methods=["GET"])
def cache_stats():
    cache = current_app.extensions.get("cache")
    if cache is None:
        return jsonify({"enabled": False})
    return jsonify({"enabled": True, "stats": cache.stats()})


@admin_bp.route("/cache", methods=["DELETE"])
def cache_clear():
    cache = current_app.extensions.get("cache")
    if cache is None:
        return jsonify({"enabled": False, "cleared": 0})
    before = cache.stats().get("size", 0) if hasattr(cache, "stats") else 0
    if hasattr(cache, "clear"):
        cache.clear()
    return jsonify({"enabled": True, "cleared": before})


@admin_bp.route("/analyzers", methods=["GET"])
def analyzers():
    registry = current_app.extensions.get("analyzer_registry")
    if registry is None:
        return jsonify({"items": []})
    return jsonify({"items": registry.names()})


@admin_bp.route("/version", methods=["GET"])
def version_endpoint():
    from .. import __version__

    return jsonify({"version": __version__})
