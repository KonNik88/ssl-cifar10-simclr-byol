from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AverageMeter:
    name: str
    fmt: str = ".4f"

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:{self.fmt}}"


class MetricTracker:
    """
    Simple dict-like tracker of AverageMeters.
    """
    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter(name)
        self.meters[name].update(value, n)

    def avg(self, name: str) -> Optional[float]:
        if name not in self.meters:
            return None
        return self.meters[name].avg

    def to_dict(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self.meters.items()}

    def reset(self) -> None:
        for m in self.meters.values():
            m.reset()
