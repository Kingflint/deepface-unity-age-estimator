from deepface_server.services.metrics import MetricsCollector


def test_metrics_records_counts_and_average():
    metrics = MetricsCollector()
    metrics.record("/analyze", 200, 100.0)
    metrics.record("/analyze", 200, 200.0)
    metrics.record("/analyze", 400, 50.0)

    snap = metrics.snapshot()
    assert snap["total_requests"] == 3
    assert snap["counts_by_path_status"]["/analyze 200"] == 2
    assert snap["counts_by_path_status"]["/analyze 400"] == 1
    # mean of 100, 200, 50 == 116.667
    assert abs(snap["avg_ms_by_path"]["/analyze"] - 116.667) < 0.01


def test_metrics_endpoint_includes_cache_stats(client):
    client.get("/healthz")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "requests" in body
    assert "cache" in body
    assert body["cache"]["max_entries"] == 8


def test_metrics_reset_clears_state():
    metrics = MetricsCollector()
    metrics.record("/x", 200, 1.0)
    metrics.reset()
    assert metrics.snapshot()["total_requests"] == 0
