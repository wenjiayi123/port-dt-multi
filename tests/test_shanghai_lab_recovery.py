from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.services.shanghai_lab_v1 import api
from app.services.shanghai_lab_v1.protocol import packed


def test_orphan_partial_file_restart_and_concurrent_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    result = json.loads((Path(__file__).parent / "fixtures/shanghai_lab_reference.json").read_text())
    packet = packed(result["snapshot"], "snapshot") | packed(result["plan"], "plan") | packed(result["energy"], "energy")
    (tmp_path / "crashed-write.tmp").write_text('{"partial":')
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(lambda _: api.execute(packet), range(8)))
    assert all(row == outputs[0] for row in outputs)
    receipt = outputs[0]
    assert receipt["restore_verified"] and not receipt["existing_demo_state_touched"]
    assert api.get_run(receipt["run_id"]) == receipt
    # A fresh lock object models a worker restart; disk receipt remains authoritative.
    import threading
    monkeypatch.setattr(api, "LOCK", threading.RLock())
    assert api.execute(packet) == receipt
    saved = api.path_for(receipt["run_id"])
    saved.write_text(json.dumps({**receipt, "restore_verified": False}))
    with pytest.raises(HTTPException) as error:
        api.execute(packet)
    assert error.value.status_code == 409
