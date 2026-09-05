from __future__ import annotations
import json
import random
from pathlib import Path
from .arrow_dataset import ArrowTrajectory

def split_settings(items: list[ArrowTrajectory], val_count: int = 1, seed: int = 42) -> tuple[list[ArrowTrajectory], list[ArrowTrajectory]]:
    groups = sorted({item.setting for item in items})
    if not groups or val_count >= len(groups):
        raise ValueError("need at least one training setting and one validation setting")
    rng = random.Random(seed); rng.shuffle(groups); val = set(groups[:val_count])
    return [x for x in items if x.setting not in val], [x for x in items if x.setting in val]


def load_official_indices(index_root: str | Path, domain: str) -> dict[str, list[dict[str, int | str]]]:
    """Load the official RealPDEBench lazy-slicing indices if present.

    ``domain`` is ``real`` or ``sim``; the latter maps to the upstream
    ``numerical`` index filename. Each returned entry identifies one window by
    its embedded trajectory id and starting ``time_id``.
    """
    root = Path(index_root)
    suffix = "real" if domain == "real" else "numerical"
    result: dict[str, list[dict[str, int | str]]] = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}_index_{suffix}.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[split] = [{"sim_id": str(item["sim_id"]), "time_id": int(item["time_id"])} for item in payload]
    return result
