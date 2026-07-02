"""M8.1 MLflow sink: the request traces become MLflow runs, idempotently."""
from mlops.mlflow_sink import log_traces


def _traces():
    return [
        {"message_id": "m1", "route": "factual", "model": "fake", "tier": "auto",
         "latency_ms": 120.0, "grounding": 0.8, "confidence": 0.6, "ts": 1.0},
        {"message_id": "m2", "route": "metric", "tier": "auto", "cost": 0.001, "ts": 2.0},
    ]


def test_log_traces_creates_runs_with_params_and_metrics(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = "file://" + str(tmp_path / "mlruns")
    result = log_traces(_traces(), tracking_uri=uri)
    assert result == {"logged": 2, "skipped": 0}

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("skein-lite")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 2
    m1 = next(r for r in runs if r.data.tags.get("trace_id") == "m1")
    assert m1.data.metrics["grounding"] == 0.8
    assert m1.data.params["route"] == "factual"


def test_log_traces_is_idempotent(tmp_path):
    uri = "file://" + str(tmp_path / "mlruns")
    log_traces(_traces(), tracking_uri=uri)
    again = log_traces(_traces(), tracking_uri=uri)
    assert again == {"logged": 0, "skipped": 2}  # already-logged runs are not duplicated


def test_non_dict_lines_and_id_less_traces_are_handled(tmp_path):
    uri = "file://" + str(tmp_path / "mlruns")
    traces = ["a bad line", {"route": "factual", "grounding": 0.5}]  # no id on the dict
    first = log_traces(traces, tracking_uri=uri)
    assert first["logged"] == 1  # the string is skipped, the dict logs with a hashed id
    again = log_traces(traces, tracking_uri=uri)
    assert again["logged"] == 0 and again["skipped"] == 1  # deduped by the hash, not re-logged
