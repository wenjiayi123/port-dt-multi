# app/infra/storage.py
"""
【大白话注释】
这是“对象存储门面”。把文件（JSON/CSV/图像/二进制）统一存起来、读出来、列出来。
现在支持：
- memory://           -> 内存字典，重启即丢（开发自测）
- file://<abs_path>   -> 本地磁盘，默认 ./data/objects

【统一路径规范（建议，便于落地与审计回放）】
- 审计证据包：audit/{event_id}.json
- 回放数据：  replay/{yyyymmdd}/{asset_id}/{name}.csv
- 报表导出：  reports/{period}/{report_name}.json
- 模型包：    models/{model_id}/artifact.bin

【时间/多租户口径】
- 若有多租户，建议把 tenant_id 放到路径前缀：{tenant_id}/audit/...
- 时间戳统一 UTC；文件名里如需时间，建议 yyyymmdd 这种无歧义格式

【真实港口落地】
- 本地 PoC/联调：使用 file://，指向挂载磁盘或 NFS
- 生产：切换到 MinIO/S3（后续在本文件里加 s3:// 实现），业务层无需改代码

【API 设计】
- save_bytes(path, content)  -> 返回可定位的 URI（例如 file:///... 或 memory://...）
- save_text/save_json        -> 语法糖
- load_bytes(uri_or_path)    -> 读回来（支持传 URI 或相对路径）
- list(prefix)               -> 列目录（前缀）
- delete(uri_or_path)        -> 删对象
- url(uri_or_path)           -> 返回“可访问URL”（本地返回 file:/// ，S3 可返回预签名URL）

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from pathlib import Path
import json
import os
import base64
import time
import datetime as dt

# ========= 工具 =========

def _ensure_bytes(data: Union[str, bytes]) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.encode("utf-8")

def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

# ========= 实现 =========

@dataclass
class StorageConfig:
    backend_url: str = ""   # "memory://", "file:///abs/path", 未来可支持 "s3://bucket?region=..."
    root: Optional[str] = None  # file 后端根目录（可选，未给时按 backend_url 解析）

class ObjectStorage:
    """
    统一对象存储门面。
    用法：
        store = ObjectStorage(StorageConfig(backend_url="file://./data/objects"))
        uri = store.save_json("audit/event-123.json", {"ok": True})
        b = store.load_bytes(uri)
        print(store.list("audit/"))
    """
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        if not self.config.backend_url:
            # 默认本地文件存储：./data/objects
            default_root = Path.cwd() / "data" / "objects"
            self.backend_url = f"file://{default_root.resolve()}"
        else:
            self.backend_url = self.config.backend_url

        # 解析文件根目录
        if self.backend_url.startswith("file://"):
            p = self.backend_url[len("file://") :]
            self._file_root = Path(p).resolve()
            self._file_root.mkdir(parents=True, exist_ok=True)
            self._mode = "file"
        elif self.backend_url.startswith("memory://"):
            self._mem = {}  # key: path -> bytes
            self._mode = "memory"
        else:
            # 未来扩展：s3:// 等
            raise ValueError(f"unsupported backend: {self.backend_url}")

    # ---- 保存 ----
    def save_bytes(self, path: str, content: Union[str, bytes]) -> str:
        """
        保存二进制内容到对象存储。
        path 是逻辑路径（相对），函数返回一个 URI（file:/// 或 memory://）
        """
        if not path or path.startswith("/"):
            raise ValueError("path 必须是相对路径，不要以 / 开头")

        if self._mode == "file":
            full = self._file_root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(_ensure_bytes(content))
            return f"file://{full.as_posix()}"
        else:
            # memory
            self._mem[path] = _ensure_bytes(content)
            return f"memory://{path}"

    def save_text(self, path: str, text: str, encoding: str = "utf-8") -> str:
        return self.save_bytes(path, text.encode(encoding))

    def save_json(self, path: str, obj: dict, ensure_ascii: bool = False, indent: int = 2) -> str:
        blob = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
        return self.save_bytes(path, blob)

    # ---- 读取 ----
    def load_bytes(self, uri_or_path: str) -> bytes:
        """
        从 URI 或 逻辑路径 读取对象内容。
        支持：
          - file:///abs/path/...
          - memory://logical/path
          - logical/path （自动按当前后端解析）
        """
        s = str(uri_or_path)
        if s.startswith("file://"):
            full = Path(s[len("file://") :])
            return full.read_bytes()
        if s.startswith("memory://"):
            key = s[len("memory://") :]
            return self._mem[key]

        # 逻辑路径
        if self._mode == "file":
            full = self._file_root / s
            return full.read_bytes()
        else:
            return self._mem[s]

    def load_text(self, uri_or_path: str, encoding: str = "utf-8") -> str:
        return self.load_bytes(uri_or_path).decode(encoding)

    def load_json(self, uri_or_path: str) -> dict:
        return json.loads(self.load_text(uri_or_path))

    # ---- 列举/删除 ----
    def list(self, prefix: str = "") -> List[str]:
        """
        列举某个前缀下的对象逻辑路径列表。
        - file:// 返回相对 root 的路径
        - memory:// 返回已存的 key 列表
        """
        pref = prefix.strip("/")
        if self._mode == "file":
            base = self._file_root
            results: List[str] = []
            if not base.exists():
                return results
            for p in base.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(base).as_posix()
                    if not pref or rel.startswith(pref):
                        results.append(rel)
            return results
        else:
            return [k for k in self._mem.keys() if (not pref) or k.startswith(pref)]

    def delete(self, uri_or_path: str) -> bool:
        """删除一个对象；不存在返回 False。"""
        s = str(uri_or_path)
        try:
            if s.startswith("file://"):
                full = Path(s[len("file://") :])
                if full.exists():
                    full.unlink()
                    return True
                return False
            if s.startswith("memory://"):
                key = s[len("memory://") :]
                return self._mem.pop(key, None) is not None

            # 逻辑路径
            if self._mode == "file":
                full = self._file_root / s
                if full.exists():
                    full.unlink()
                    return True
                return False
            else:
                return self._mem.pop(s, None) is not None
        except Exception:
            return False

    # ---- URL/定位 ----
    def url(self, uri_or_path: str) -> str:
        """
        返回“可访问URL”。本地返回 file:/// 开头；memory 返回 memory://。
        上生产后可在 s3:// 实现里返回“预签名URL”。
        """
        s = str(uri_or_path)
        if s.startswith(("file://", "memory://")):
            return s
        if self._mode == "file":
            full = (self._file_root / s).resolve()
            return f"file://{full.as_posix()}"
        else:
            return f"memory://{s}"

# ========= 冒烟测试 =========

def _smoke() -> dict:
    """
    演示：
      1) 保存一个审计证据包 audit/event-<ts>.json
      2) 列举 audit/ 前缀
      3) 读回来并校验
    """
    store = ObjectStorage(StorageConfig(backend_url="file://./data/objects"))
    eid = f"evt-{int(time.time())}"
    path = f"audit/{eid}.json"
    payload = {"event_id": eid, "ts": _now_utc_iso(), "ok": True}

    uri = store.save_json(path, payload)
    listed = store.list("audit/")
    loaded = store.load_json(uri)
    ok = (loaded.get("event_id") == eid)

    return {"uri": uri, "listed_contains": any(path in x for x in listed), "ok": ok}

if __name__ == "__main__":
    import json
    print(json.dumps(_smoke(), ensure_ascii=False, indent=2))
