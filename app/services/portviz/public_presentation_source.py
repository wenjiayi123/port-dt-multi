"""PortViz view of the cited terminal presentation contract, not measured tracks."""
from __future__ import annotations
import time
from typing import Any

from app.services.public_port_demo.service import profile, snapshot


class PublicPresentationSource:
    def __init__(self) -> None:
        p = profile(); f = p["facts"]
        self.layout: dict[str, Any] = {
            "world": {"W": 1600, "H": 900},
            "lanes": [[{"x": 120, "y": y}, {"x": 1480, "y": y}] for y in (640, 700, 760)]
                + [[{"x": x, "y": 560}, {"x": x, "y": 780}] for x in (250, 760, 1280)],
            "yards": [{"x": 130 + c * 131, "y": 220 + r * 61, "w": 118, "h": 48} for r in range(6) for c in range(10)],
            "berth": {"x": 100, "y": 60, "w": 1400, "h": 24, "count": f["berths"]["value"]},
            "qcs": [{"x": 125 + i * 47.7, "y": 110} for i in range(f["qc_count"]["value"])],
            "ycs": [{"x": 165 + (i % 11) * 120, "y": 245 + (i // 11) * 31} for i in range(f["rmg_count"]["value"])],
            "meta": {"port_code": "CNSHA", "port_name_zh": p["name"], "timezone": p["timezone"],
                     "source_mode": "public_parameter_calibrated_demo", "profile_id": p["profile_id"],
                     "profile_sha256": p["sha256"], "measured_entity_tracks": False,
                     "geometry_class": "schematic_arrangement_public_dimensions_not_surveyed_positions",
                     "published_parameters": f, "provenance": p["boundary"]},
        }

    def get_bootstrap(self) -> dict[str, Any]:
        return self.layout

    def next_frame(self, since: int | None = None) -> dict[str, Any]:
        del since
        s = snapshot(); o = s["operating"]; tick = time.monotonic()
        expected = {"berths_total": 7, "qc_total": 29, "rmg_total": 121, "agv_total": 155}
        if any(o.get(key) != value for key, value in expected.items()):
            raise ValueError("public_demo layout is Yangshan IV schematic; use PORTVIZ_MODE=real and a site layout for different resource counts")
        working = [c for c in s["calls"] if c["phase"] == "working"]
        return {
            "ts": int(time.time() * 1000), "source_timestamp": s["source_timestamp"],
            "agv": [{"lane": i % len(self.layout["lanes"]), "s": (i * 13 + tick * .15) % 100,
                     "alarm": False, "busy": i < o["agv_busy"]} for i in range(o["agv_total"])],
            "qc": [{"busy": i < o["qc_busy"], "trolley": (i * .11 + tick * .015) % 1} for i in range(o["qc_total"])],
            "yc": [{"busy": i < o["rmg_busy"]} for i in range(o["rmg_total"])],
            "tr": [], "hotspots": [],
            "vessels": [{"berth": c["berth_index"], "progress": c["progress"], "len": 250,
                         "call_id": c["call_id"], "geometry_class": "generic_ship_icon_not_actual_LOA"} for c in working],
            "metrics": {"base_load_kw": o["power_kw"], "throughput_teu": o["throughput_teu_per_hour"],
                        "vessel_arrivals": o["working_call_count"], "berths_busy": o["berths_busy"],
                        "berths_total": o["berths_total"], "berth_occupancy_pct": o["berth_occupancy_pct"]},
            "meta": {"source_mode": s["mode"], "profile_id": s["profile"]["profile_id"],
                     "measured_entity_tracks": False, "simulation_mode": True, "production_authority": False,
                     "positions": "visual_projection_not_AIS", "vessel_length": "icon_scale_not_actual_ship_dimension"},
        }
