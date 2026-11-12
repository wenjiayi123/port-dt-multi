"""Serialize the same UTC hours for three controllers, rejecting incomplete native rollouts."""
import math


METHODS = ("idle", "selected", "best_fixed")
FIELDS = ("hour", "pcc_kw", "bess_kw", "flex_kw", "soc", "mandatory_shore_kw", "option")


def scene_frames(methods):
    result = {}
    for name in METHODS:
        series = methods[name]["hours"]
        if len(series) != 168:
            raise ValueError("incomplete native week")
        rows = []
        for expected, row in enumerate(series):
            frame = {key: row[key] for key in FIELDS}
            if type(frame["hour"]) is not int or frame["hour"] != expected or not isinstance(frame["option"], str):
                raise ValueError("native frame identity mismatch")
            if any(type(frame[k]) not in (int, float) or not math.isfinite(frame[k]) for k in FIELDS[1:-1]):
                raise ValueError("invalid physical reading")
            if not 0 <= frame["soc"] <= 1 or frame["mandatory_shore_kw"] < 0:
                raise ValueError("invalid plant state")
            rows.append(frame)
        result[name] = rows
    return result
