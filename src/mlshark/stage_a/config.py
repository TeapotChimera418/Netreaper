from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StageAConfig:
    data_path: Path
    target_column: str
    model_name: str = "random_forest"
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    output_dir: Path = Path("artifacts/stage_a")
    model_params: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if not 0.0 <= self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1")
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be less than 1")
