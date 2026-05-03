"""POST /jobs and GET /jobs/{id}."""
from flask import Blueprint, current_app, jsonify, request

jobs_bp = Blueprint("jobs", __name__)


@jobs_bp.route("/jobs", methods=["POST"])
def submit_job():
    manager = current_app.extensions.get("job_manager")
    if manager is None:
        return jsonify({"error": "jobs disabled", "code": "jobs_disabled"}), 503
    payload = request.get_json(silent=True) or {}
    kind = payload.get("kind", "analyze")
    job = manager.submit(kind, payload.get("payload") or {})
    return jsonify(job.to_dict()), 202


@jobs_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    manager = current_app.extensions.get("job_manager")
    if manager is None:
        return jsonify({"error": "jobs disabled", "code": "jobs_disabled"}), 503
    job = manager.get(job_id)
    if job is None:
        return jsonify({"error": "not found", "code": "not_found"}), 404
    return jsonify(job.to_dict())


@jobs_bp.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    manager = current_app.extensions.get("job_manager")
    if manager is None:
        return jsonify({"error": "jobs disabled", "code": "jobs_disabled"}), 503
    if not manager.cancel(job_id):
        return jsonify({"error": "cannot cancel", "code": "cannot_cancel"}), 409
    return jsonify({"id": job_id, "status": "cancelled"})


@jobs_bp.route("/jobs", methods=["GET"])
def list_jobs():
    manager = current_app.extensions.get("job_manager")
    if manager is None:
        return jsonify({"items": [], "enabled": False})
    return jsonify(
        {
            "items": [j.to_dict() for j in manager.queue.all()],
            "stats": manager.stats(),
            "enabled": True,
        }
    )
