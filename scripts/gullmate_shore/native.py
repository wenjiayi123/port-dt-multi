"""Read the frozen TwinLab benchmark without starting its web service."""
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import torch

WORK=Path(__file__).resolve().parent
DEPLOY=Path("G:/AI/WindowsDeploy")
TWIN=DEPLOY/"apps/twin"
ROOT=Path("G:/AI/Runs/gullmate-next-20261106/shore-storage")
sys.path.insert(0,str(TWIN))
from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
from scripts import train_shore_bess_v8 as train

torch.set_num_threads(1)
PERIODS={"train":("2024-01-01","2025-05-01"),"validation":("2025-05-01","2025-08-01"),
         "repeat":("2025-08-01","2026-01-01"),"forward":("2026-01-01","2026-06-01")}


def read(p):return json.loads(p.read_bytes())
def sha(p):return sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+"."+uuid4().hex+".part")
    try:
        tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
        os.replace(tmp,p)
    finally:tmp.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def history():
    cfg=train.load_config(train.CONFIG_PATH)
    dataset=train.load_port_dataset(cfg["dataset_id"],data_root=TWIN/"data/rl/datasets")
    return cfg,dataset,train.month_slice(dataset,*PERIODS["train"])


def plant(split,**kwargs):
    cfg,data,scale=history()
    if split=="forward":
        data=train.joined_forward(data,train.load_port_dataset("public_cn_sha_forward_2026m05_v1",data_root=TWIN/"data/rl/datasets"))
    selected=train.month_slice(data,*PERIODS[split])
    return ShoreBESSV8SACEnv(data,selected,config=cfg,normalization_slice=scale,episode_steps=168,carbon_price=16.,
                             training=False,seed=0,record_trace=False,**kwargs)


def starts(split):
    env=plant(split)
    try:return train.window_starts(len(env.segment))
    finally:env.close()
