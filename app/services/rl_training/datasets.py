from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


CANONICAL_COLUMNS = (
    "timestamp",
    "base_load_kw",
    "throughput_teu",
    "vessel_arrivals",
    "tide_m",
    "price_per_kwh",
    "carbon_kg_per_kwh",
    "ambient_c",
)
NUMERIC_COLUMNS = CANONICAL_COLUMNS[1:]
DEFAULT_DATA_ROOT = Path("data/rl/datasets")
COLUMN_UNITS = {
    "base_load_kw": "kW",
    "throughput_teu": "TEU/hour",
    "vessel_arrivals": "count/hour",
    "tide_m": "m",
    "price_per_kwh": "currency/kWh",
    "carbon_kg_per_kwh": "kgCO2e/kWh",
    "ambient_c": "degreeCelsius",
}
PHYSICAL_BOUNDS = {
    "base_load_kw": (0.0, None),
    "throughput_teu": (0.0, None),
    "vessel_arrivals": (0.0, None),
    "tide_m": (-20.0, 20.0),
    "price_per_kwh": (0.0, None),
    "carbon_kg_per_kwh": (0.0, 5.0),
    "ambient_c": (-60.0, 70.0),
}
REQUIRED_GOVERNANCE_METADATA = ("provenance_type", "license", "owner", "timezone", "intended_use")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_dataset_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    if not cleaned or len(cleaned) > 64:
        raise ValueError("dataset_id must contain 1-64 letters, numbers, or separators")
    return cleaned


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PortDataset:
    dataset_id: str
    path: Path
    timestamps: Sequence[str]
    values: np.ndarray
    metadata: Dict[str, Any]

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def fingerprint(self) -> str:
        return str(self.metadata.get("sha256") or file_sha256(self.path))

    def split(self, test_ratio: float = 0.2) -> tuple[slice, slice]:
        ratio = min(0.4, max(0.1, float(test_ratio)))
        split_at = max(2, min(self.rows - 2, int(self.rows * (1.0 - ratio))))
        return slice(0, split_at), slice(split_at, self.rows)

    def describe(self, test_ratio: float = 0.2) -> Dict[str, Any]:
        train_slice, test_slice = self.split(test_ratio)
        return {
            **self.metadata,
            "dataset_id": self.dataset_id,
            "artifact_id": self.path.name,
            "rows": self.rows,
            "columns": list(CANONICAL_COLUMNS),
            "train_rows": train_slice.stop - train_slice.start,
            "test_rows": test_slice.stop - test_slice.start,
            "split_method": "chronological_holdout_no_shuffle",
            "test_ratio": float(test_ratio),
            "quality": dataset_quality_report(self),
        }


def dataset_quality_report(dataset: PortDataset) -> Dict[str, Any]:
    """Return a deterministic, auditable quality gate for one canonical dataset."""
    timestamps = [_timestamp_value(value) for value in dataset.timestamps]
    gaps = np.asarray(
        [(timestamps[index] - timestamps[index - 1]).total_seconds() for index in range(1, len(timestamps))],
        dtype=np.float64,
    )
    cadence_sec = float(np.median(gaps)) if gaps.size else 0.0
    cadence_tolerance = max(1.0, cadence_sec * 0.01)
    irregular_gaps = int(np.sum(np.abs(gaps - cadence_sec) > cadence_tolerance)) if gaps.size else 0
    columns: Dict[str, Any] = {}
    physical_violations = 0
    constant_columns: List[str] = []
    for index, column in enumerate(NUMERIC_COLUMNS):
        values = dataset.values[:, index].astype(np.float64)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = float(q3 - q1)
        lower_fence = float(q1 - 1.5 * iqr)
        upper_fence = float(q3 + 1.5 * iqr)
        outlier_count = int(np.sum((values < lower_fence) | (values > upper_fence))) if iqr > 0 else 0
        lower, upper = PHYSICAL_BOUNDS[column]
        violations = int(np.sum(values < lower)) if lower is not None else 0
        violations += int(np.sum(values > upper)) if upper is not None else 0
        physical_violations += violations
        is_constant = bool(np.ptp(values) <= 1e-12)
        if is_constant:
            constant_columns.append(column)
        columns[column] = {
            "unit": COLUMN_UNITS[column],
            "min": float(np.min(values)),
            "p25": float(q1),
            "median": float(median),
            "p75": float(q3),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "outlier_count_iqr": outlier_count,
            "physical_bounds": {"min": lower, "max": upper},
            "physical_violation_count": violations,
            "constant": is_constant,
        }
    missing_metadata = [name for name in REQUIRED_GOVERNANCE_METADATA if not dataset.metadata.get(name)]
    errors: List[str] = []
    warnings: List[str] = []
    if physical_violations:
        errors.append(f"{physical_violations} values violate canonical physical bounds")
    if cadence_sec <= 0:
        errors.append("sampling cadence cannot be determined")
    if irregular_gaps:
        warnings.append(f"{irregular_gaps} timestamp gaps differ from the median cadence")
    if constant_columns:
        warnings.append("constant columns: " + ", ".join(constant_columns))
    if missing_metadata:
        errors.append("missing governance metadata: " + ", ".join(missing_metadata))
    status = "fail" if errors else ("warn" if warnings else "pass")
    return {
        "status": status,
        "training_eligible": not errors,
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.fingerprint,
        "rows": dataset.rows,
        "time": {
            "start_at": dataset.timestamps[0],
            "end_at": dataset.timestamps[-1],
            "median_cadence_seconds": cadence_sec,
            "irregular_gap_count": irregular_gaps,
            "strictly_increasing": True,
        },
        "missing_value_count": 0,
        "non_finite_value_count": 0,
        "physical_violation_count": physical_violations,
        "missing_governance_metadata": missing_metadata,
        "columns": columns,
        "errors": errors,
        "warnings": warnings,
        "generated_at": utc_now(),
    }


def _finite_number(raw: Any, column: str, line: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: {column} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"line {line}: {column} must be finite")
    return value


def _parse_timestamp(raw: Any, line: int) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"line {line}: timestamp is empty")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"line {line}: timestamp must be ISO-8601: {text!r}") from exc
    return text


def _timestamp_value(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_port_dataset(dataset_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> PortDataset:
    resolved = safe_dataset_id(dataset_id)
    path = data_root / f"{resolved}.csv"
    meta_path = data_root / f"{resolved}.meta.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {resolved}")
    timestamps: List[str] = []
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in CANONICAL_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"dataset {resolved} missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, 2):
            timestamp = _parse_timestamp(row.get("timestamp"), line)
            if timestamps and _timestamp_value(timestamp) <= _timestamp_value(timestamps[-1]):
                raise ValueError(f"line {line}: timestamps must be strictly increasing")
            timestamps.append(timestamp)
            rows.append([_finite_number(row.get(column), column, line) for column in NUMERIC_COLUMNS])
    if len(rows) < 48:
        raise ValueError(f"dataset {resolved} needs at least 48 chronological rows; got {len(rows)}")
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "dataset_id": resolved,
            "sha256": file_sha256(path),
            "rows": len(rows),
            "start_at": timestamps[0],
            "end_at": timestamps[-1],
        }
    )
    return PortDataset(resolved, path, timestamps, np.asarray(rows, dtype=np.float32), metadata)


def list_datasets(data_root: Path = DEFAULT_DATA_ROOT) -> List[Dict[str, Any]]:
    data_root.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for path in sorted(data_root.glob("*.csv")):
        try:
            items.append(load_port_dataset(path.stem, data_root).describe())
        except Exception as exc:
            items.append({"dataset_id": path.stem, "artifact_id": path.name, "valid": False, "error": str(exc)})
    return items


def import_dataset(
    source_path: Path,
    dataset_id: str,
    mapping: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Dict[str, Any]:
    """Import a port export through an explicit canonical-column mapping.

    ``mapping`` maps canonical names to source CSV names. No values are silently
    invented here: every canonical field must be present after mapping.
    """
    resolved = safe_dataset_id(dataset_id)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(str(source_path))
    data_root.mkdir(parents=True, exist_ok=True)
    supplied_metadata = dict(metadata or {})
    missing_governance = [name for name in ("license", "owner", "timezone", "intended_use") if not supplied_metadata.get(name)]
    if missing_governance:
        raise ValueError("metadata missing required governance fields: " + ", ".join(missing_governance))
    mapping = {name: str((mapping or {}).get(name, name)) for name in CANONICAL_COLUMNS}
    target = data_root / f"{resolved}.csv"
    meta_path = data_root / f"{resolved}.meta.json"
    replace_existing = supplied_metadata.pop("replace_existing", False) is True
    if (target.exists() or meta_path.exists()) and not replace_existing:
        raise FileExistsError(f"dataset already exists: {resolved}; set metadata.replace_existing=true explicitly to replace it")
    tmp = data_root / f".{resolved}.importing.csv"
    row_count = 0
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as src, tmp.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            reader = csv.DictReader(src)
            missing = [source for source in mapping.values() if source not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"source CSV missing mapped columns: {', '.join(missing)}")
            writer = csv.DictWriter(dst, fieldnames=CANONICAL_COLUMNS)
            writer.writeheader()
            previous_timestamp: Optional[str] = None
            for line, row in enumerate(reader, 2):
                output = {canonical: row[source] for canonical, source in mapping.items()}
                timestamp = _parse_timestamp(output["timestamp"], line)
                if previous_timestamp and _timestamp_value(timestamp) <= _timestamp_value(previous_timestamp):
                    raise ValueError(f"line {line}: timestamps must be strictly increasing")
                previous_timestamp = timestamp
                for column in NUMERIC_COLUMNS:
                    value = _finite_number(output[column], column, line)
                    lower, upper = PHYSICAL_BOUNDS[column]
                    if (lower is not None and value < lower) or (upper is not None and value > upper):
                        raise ValueError(f"line {line}: {column}={value} violates physical bounds [{lower}, {upper}]")
                writer.writerow(output)
                row_count += 1
        if row_count < 48:
            raise ValueError(f"dataset {resolved} needs at least 48 chronological rows; got {row_count}")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    meta = {
        "dataset_id": resolved,
        "created_at": utc_now(),
        "provenance_type": "user_supplied_port_export",
        "mapping": mapping,
        "source_filename": source_path.name,
        **supplied_metadata,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_port_dataset(resolved, data_root).describe()


def write_canonical_rows(
    dataset_id: str,
    rows: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Dict[str, Any]:
    resolved = safe_dataset_id(dataset_id)
    data_root.mkdir(parents=True, exist_ok=True)
    path = data_root / f"{resolved}.csv"
    tmp = data_root / f".{resolved}.building.csv"
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        previous_timestamp: Optional[str] = None
        for line, row in enumerate(rows, 2):
            output = {name: row[name] for name in CANONICAL_COLUMNS}
            timestamp = _parse_timestamp(output["timestamp"], line)
            if previous_timestamp and _timestamp_value(timestamp) <= _timestamp_value(previous_timestamp):
                raise ValueError(f"line {line}: timestamps must be strictly increasing")
            previous_timestamp = timestamp
            for column in NUMERIC_COLUMNS:
                _finite_number(output[column], column, line)
            writer.writerow(output)
    tmp.replace(path)
    (data_root / f"{resolved}.meta.json").write_text(
        json.dumps({"dataset_id": resolved, **dict(metadata)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return load_port_dataset(resolved, data_root).describe()
