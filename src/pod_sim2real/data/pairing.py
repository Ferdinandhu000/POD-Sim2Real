from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .arrow_dataset import ArrowTrajectory

@dataclass(frozen=True)
class TrajectoryPair:
    key: tuple[int, float]
    real: ArrowTrajectory | None
    sim: ArrowTrajectory | None
    exact_id: bool

def pair_trajectories(real: Iterable[ArrowTrajectory], sim: Iterable[ArrowTrajectory]) -> list[TrajectoryPair]:
    real_map = {t.setting: t for t in real}; sim_map = {t.setting: t for t in sim}
    keys = sorted(set(real_map) | set(sim_map))
    return [TrajectoryPair(k, real_map.get(k), sim_map.get(k), bool(real_map.get(k) and sim_map.get(k) and real_map[k].sim_id == sim_map[k].sim_id)) for k in keys]
