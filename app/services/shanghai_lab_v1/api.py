"""Durable isolated trial execution and receipt readback; no production mutation."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import threading
import tempfile
from app.services.durable_file import publish
from app.services.file_lock import exclusive_lock
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from .data import digest
from .protocol import inputs, unpack, verify_energy

router = APIRouter(prefix="/api/shanghai-lab/v1", tags=["Shanghai-public-lab"])
ROOT = Path(__file__).resolve().parents[3] / ".runtime/shanghai-lab-v1"
LOCK = threading.RLock()


@contextmanager
def execution_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK, exclusive_lock(ROOT / ".execution.lock"):
        yield


def path_for(run_id):
    if not re.fullmatch(r"sha-[a-f0-9]{24}", run_id):
        raise ValueError("invalid run id")
    return ROOT / (run_id + ".json")


@router.get("/health")
def health():
    return {"schema": "shanghai-adapter.v1", "system": "port-dt-multi", "production_authority": False}


@router.post("/execute")
def execute(payload: dict):
    try:
        case, plan, energy = inputs(payload), unpack(payload, "plan"), unpack(payload, "energy")
        ledger = verify_energy(case, plan, energy)
        run_id = "sha-" + digest(payload)[:24]
        file = path_for(run_id)
        with execution_lock():
            if file.exists():
                return get_run(run_id)
            initial = {"job_states": {j["id"]: "planned" for j in case["jobs"]}, "completed_teu": 0,
                       "stored_kwh": case["battery"]["initial_soc"] * case["battery"]["capacity_kwh"]}
            state = json.loads(json.dumps(initial))
            events = []
            epoch = datetime.fromisoformat(case["event_time"])
            for h in range(24):
                for row in plan["scheduled"]:
                    if row["start_hour"] == h:
                        if state["job_states"][row["job_id"]] != "planned":
                            raise ValueError("invalid start transition")
                        state["job_states"][row["job_id"]] = "running"
                    if row["end_hour"] == h + 1:
                        if state["job_states"][row["job_id"]] != "running":
                            raise ValueError("invalid completion transition")
                        state["job_states"][row["job_id"]] = "completed"
                state["completed_teu"] += ledger["operations"]["completed_teu"][h]
                state["stored_kwh"] = ledger["hourly"][h]["stored_kwh"]
                event = {"seq": h + 1, "event_time": (epoch + timedelta(hours=h+1)).isoformat(),
                         "state_sha256": digest(state), "state": json.loads(json.dumps(state)),
                         "previous_sha256": events[-1]["sha256"] if events else "0"*64}
                event["sha256"] = digest(event)
                events.append(event)
            if any(v != "completed" for v in state["job_states"].values()):
                raise ValueError("partial execution")
            final = json.loads(json.dumps(state))
            state = json.loads(json.dumps(initial))  # dispose isolated scenario state, preserve final receipt
            result = {"schema": "shanghai-execution-receipt.v1", "run_id": run_id, "trial_id": case["trial_id"],
                      "port_id": case["port_id"], "source_snapshot_sha256": case["snapshot_sha256"],
                      "dataset_sha256": case["dataset_sha256"], "status": "completed_simulation_restored",
                      "events": events, "initial_state_sha256": digest(initial), "final_state": final,
                      "restored_state_sha256": digest(state), "restore_verified": digest(state) == digest(initial),
                      "totals": ledger["totals"], "created_at": datetime.now(timezone.utc).isoformat(),
                      "production_authority": False, "existing_demo_state_touched": False}
            result["receipt_sha256"] = digest(result)
            ROOT.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", prefix=run_id+"-", suffix=".tmp", dir=ROOT, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            publish(temporary, file)
            return result
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    try:
        result = json.loads(path_for(run_id).read_text())
        if digest({k: v for k, v in result.items() if k != "receipt_sha256"}) != result["receipt_sha256"]:
            raise ValueError("stored receipt integrity mismatch")
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, "trial receipt not found") from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
