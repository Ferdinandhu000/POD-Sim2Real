from __future__ import annotations
import json, logging
from pathlib import Path

class JsonlHandler(logging.Handler):
    def __init__(self,path): super().__init__(); self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def emit(self,record):
        payload=getattr(record,"metrics",None)
        if payload is not None:
            with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(payload,ensure_ascii=True)+"\n")

def make_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True,exist_ok=True); logger=logging.getLogger(f"pod_sim2real.{log_dir}"); logger.setLevel(logging.INFO); logger.handlers.clear(); logger.propagate=False
    formatter=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream=logging.StreamHandler(); stream.setFormatter(formatter); logger.addHandler(stream)
    file=logging.FileHandler(log_dir/"train.log",encoding="utf-8"); file.setFormatter(formatter); logger.addHandler(file)
    error=logging.FileHandler(log_dir/"error.log",encoding="utf-8"); error.setLevel(logging.ERROR); error.setFormatter(formatter); logger.addHandler(error)
    logger.addHandler(JsonlHandler(log_dir.parent/"metrics.jsonl")); return logger

def log_metrics(logger, metrics): logger.info("epoch metrics", extra={"metrics":metrics})
