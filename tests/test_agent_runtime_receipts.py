import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.services.agent_runtime_receipts import (
    AgentRuntimeReceipts,
    StaleContext,
    build_router,
)


class Telemetry:
    sha = "a" * 64

    def source_status(self):
        return {"sha256": self.sha, "production_authority": False}

    def agent_snapshot_context(self):
        return {
            "replay_anchor_position": 123.45,
            "source": self.source_status(),
            "production_authority": False,
        }


class Runtime:
    def __init__(self):
        self.calls = []
        self.after_calculation = lambda: None

    def status(self):
        return {
            "available": True,
            "model": {"loaded_model_sha256": "b" * 64, "model_sha256": "c" * 64},
        }

    def coverage(self):
        return {"scenarios": [{"id": "high_density_berthing", "state": "supported"}]}

    def series(self, **kwargs):
        self.calls.append(kwargs)
        self.after_calculation()
        return {
            "available": True,
            "policy": self.status()["model"],
            "production_authority": False,
            "series": [{"queue": 2}],
        }


@pytest.fixture
def fixture(tmp_path):
    runtime, telemetry, now = Runtime(), Telemetry(), [1000.0]
    service = AgentRuntimeReceipts(runtime, telemetry, tmp_path, clock=lambda: now[0])
    context = service.capture()
    payload = {
        "request_id": "test-transaction-001",
        "context_id": context["context_id"],
        "scenario": "high_density_berthing",
        "horizon_min": 60,
        "step_min": 15,
    }
    return service, runtime, telemetry, now, payload


def test_response_loss_restart_duplicate_and_pinned_input(fixture):
    service, runtime, telemetry, now, payload = fixture
    first = service.evaluate(payload)["receipt"]
    restored = AgentRuntimeReceipts(
        runtime, telemetry, service.root, clock=lambda: now[0]
    )
    assert restored.get(payload["request_id"]) == first
    assert restored.evaluate(payload) == {"duplicate": True, "receipt": first}
    assert runtime.calls == [
        {
            "horizon_min": 60,
            "step_min": 15,
            "scenario": "high_density_berthing",
            "replay_anchor_position": 123.45,
        }
    ]
    assert (
        first["source_context"]["loaded_model_sha256"]
        != first["source_context"]["declared_model_sha256"]
    )
    with pytest.raises(StaleContext, match="different calculation"):
        restored.evaluate({**payload, "horizon_min": 30})


@pytest.mark.parametrize("change", ["time", "source", "during"])
def test_stale_before_and_during_calculation_rejected(fixture, change):
    service, runtime, telemetry, now, payload = fixture
    if change == "time":
        now[0] += 301
    elif change == "source":
        telemetry.sha = "d" * 64
    else:
        runtime.after_calculation = lambda: setattr(telemetry, "sha", "d" * 64)
    with pytest.raises(StaleContext):
        service.evaluate(payload)
    assert service.get(payload["request_id"]) is None
    assert len(runtime.calls) == (1 if change == "during" else 0)


def test_concurrent_duplicate_computes_once(fixture):
    service, runtime, _, _, payload = fixture
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda _: service.evaluate(payload), range(2)))
    assert len(runtime.calls) == 1
    assert sorted(row["duplicate"] for row in rows) == [False, True]
    assert rows[0]["receipt"] == rows[1]["receipt"]


def test_receipt_and_context_tamper_rejected(fixture):
    service, _, _, _, payload = fixture
    service.evaluate(payload)
    path = service.root / "receipts" / (payload["request_id"] + ".json")
    value = json.loads(path.read_text())
    value["production_authority"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        service.get(payload["request_id"])
    path = service.root / "contexts" / (payload["context_id"] + ".json")
    value = json.loads(path.read_text())
    value["snapshot"]["replay_anchor_position"] = 999
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        service.context(payload["context_id"])


def test_http_recovery_reports_staleness_and_blocks_browser(fixture):
    service, _, _, now, payload = fixture
    app = FastAPI()
    app.include_router(build_router(service))

    @app.middleware("http")
    async def local_test_transport(request, call_next):
        request.scope["client"] = ("127.0.0.1", 1234)
        return await call_next(request)

    client = TestClient(app)
    base = "/api/v3/runtime/agent"
    assert client.post(base + "/evaluations", json=payload).status_code == 200
    now[0] += 301
    response = client.get(base + "/evaluations/" + payload["request_id"])
    assert response.status_code == 200
    assert response.json()["context_current"] is False
    assert (
        client.post(
            base + "/evaluations",
            json={**payload, "request_id": "another-test-request"},
        ).status_code
        == 409
    )
    assert (
        client.get(
            base + "/context", headers={"Origin": "https://example.org"}
        ).status_code
        == 403
    )
    assert client.get(base + "/evaluations/invalid").status_code == 422
