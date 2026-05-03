"""GET /history endpoint exposing recent analysis records."""
from flask import Blueprint, current_app, jsonify, request

history_bp = Blueprint("history", __name__)


@history_bp.route("/history", methods=["GET"])
def list_history():
    repo = current_app.extensions.get("analysis_repository")
    if repo is None:
        return jsonify({"items": [], "enabled": False})
    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 200))
    except ValueError:
        limit = 20
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        offset = 0
    items = repo.list(limit=limit, offset=offset)
    return jsonify(
        {
            "items": [r.to_dict() for r in items],
            "limit": limit,
            "offset": offset,
            "total": repo.count(),
            "enabled": True,
        }
    )


@history_bp.route("/history/<int:record_id>", methods=["GET"])
def get_record(record_id: int):
    repo = current_app.extensions.get("analysis_repository")
    if repo is None:
        return jsonify({"error": "history disabled", "code": "history_disabled"}), 404
    record = repo.get(record_id)
    if record is None:
        return jsonify({"error": "not found", "code": "not_found"}), 404
    return jsonify(record.to_dict())
