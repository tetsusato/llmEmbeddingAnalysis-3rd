from dataclasses import dataclass
from typing import Callable
import optuna

@dataclass(frozen=True)
class Mode:
    name: str
    proc: Callable[[optuna.trial.Trial], list[float]]
    hyper_params: list[str]

