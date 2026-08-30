"""Inventory repository UI controls and fail on unbound static buttons.

Runtime browser acceptance remains a separate gate. This scanner catches the
opposite class of defect: a visible button shipped without an ID/data contract,
inline handler, form-submit role or referenced class-based delegated handler.
"""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "app/ui/index.html",
    "app/ui/integration_hub.html",
    "app/ui/ops_copilot.html",
    "app/ui/v3/index.html",
    "app/ui/v3/v3.js",
    "app/ui/rl_future/rl_future_panel.html",
    "app/ui/rl_future/rl_future.js",
    "app/server.py",
)
IGNORED_CLASS_TOKENS = {
    "active",
    "btn",
    "button",
    "ghost",
    "primary",
    "secondary",
    "small",
}


class ButtonParser(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.rows: list[dict[str, Any]] = []
        self._button: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "button":
            return
        values = {str(name).lower(): str(value or "") for name, value in attrs}
        self._button = {
            "source": self.source,
            "line": self.getpos()[0],
            "id": values.get("id") or None,
            "type": values.get("type") or None,
            "onclick": values.get("onclick") or None,
            "disabled": "disabled" in values,
            "classes": [item for item in values.get("class", "").split() if item],
            "data_attributes": {
                key: value for key, value in values.items() if key.startswith("data-")
            },
            "text": "",
        }

    def handle_data(self, data: str) -> None:
        if self._button is not None:
            self._button["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "button" and self._button is not None:
            self._button["text"] = re.sub(r"\s+", " ", self._button["text"]).strip()
            self.rows.append(self._button)
            self._button = None


def _referenced(row: dict[str, Any], corpus: str) -> tuple[bool, str]:
    if row["disabled"]:
        return True, "intentionally_disabled_fail_closed"
    if row["onclick"]:
        return True, "inline_handler"
    if (row["type"] or "").lower() == "submit":
        return True, "form_submit"
    identifier = row["id"]
    if identifier:
        patterns = (
            rf"getElementById\(\s*['\"]{re.escape(identifier)}['\"]",
            rf"(?:byId|\$)\(\s*['\"]#?{re.escape(identifier)}['\"]",
            rf"querySelector(?:All)?\(\s*['\"][^'\"]*#{re.escape(identifier)}(?:[^A-Za-z0-9_-]|['\"])",
            rf"#{re.escape(identifier)}\b[^\n]{{0,200}}addEventListener",
            rf"closest(?:\?\.)?\(\s*['\"][^'\"]*#{re.escape(identifier)}(?:[^A-Za-z0-9_-]|['\"])",
            rf"\.id\s*===?\s*['\"]{re.escape(identifier)}['\"]",
            rf"['\"]{re.escape(identifier)}['\"]\s*:\s*(?:async\s*)?\(",
        )
        if any(re.search(pattern, corpus) for pattern in patterns):
            return True, "id_listener_or_reference"
    for name in row["data_attributes"]:
        stem = name.removeprefix("data-")
        camel = stem.split("-")[0] + "".join(
            part[:1].upper() + part[1:] for part in stem.split("-")[1:]
        )
        patterns = (
            rf"\[{re.escape(name)}(?:[=\]])",
            rf"dataset\.{re.escape(camel)}\b",
            rf"getAttribute\(\s*['\"]{re.escape(name)}['\"]",
        )
        if any(re.search(pattern, corpus) for pattern in patterns):
            return True, f"delegated_{name}"
    for token in row["classes"]:
        if token in IGNORED_CLASS_TOKENS:
            continue
        if re.search(rf"[.]{re.escape(token)}\b", corpus):
            return True, f"class_handler_{token}"
    return False, "unbound"


def audit() -> dict[str, Any]:
    texts = {
        source: (ROOT / source).read_text(encoding="utf-8") for source in SOURCES
    }
    corpus = "\n".join(texts.values())
    rows: list[dict[str, Any]] = []
    for source, text in texts.items():
        parser = ButtonParser(source)
        parser.feed(text)
        for row in parser.rows:
            bound, mechanism = _referenced(row, corpus)
            rows.append({**row, "bound": bound, "binding_mechanism": mechanism})
    unresolved = [row for row in rows if not row["bound"]]
    by_source = {
        source: {
            "button_count": sum(row["source"] == source for row in rows),
            "bound_count": sum(
                row["source"] == source and row["bound"] for row in rows
            ),
            "unresolved_count": sum(
                row["source"] == source and not row["bound"] for row in rows
            ),
        }
        for source in SOURCES
    }
    return {
        "schema": "port-dt-ui-control-static-audit.v1",
        "status": "PASS" if not unresolved else "FAIL",
        "source_count": len(SOURCES),
        "static_button_definition_count": len(rows),
        "bound_button_definition_count": sum(row["bound"] for row in rows),
        "unresolved_button_definition_count": len(unresolved),
        "by_source": by_source,
        "unresolved": unresolved,
        "boundary": {
            "static_binding_audit_only": True,
            "runtime_browser_acceptance_required": True,
            "async_final_receipts_required": True,
            "production_authority": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
