from __future__ import annotations

from typing import Dict


class Callback:
    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        pass


class EarlyStopping(Callback):
    """Early Stopping when monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == "min":
            improved = (self.best - current) > self.min_delta
        else:
            improved = (current - self.best) > self.min_delta

        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
