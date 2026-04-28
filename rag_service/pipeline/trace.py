from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PipelineStep:
    name: str
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


class PipelineTrace:
    def __init__(self) -> None:
        self.steps: list[PipelineStep] = []

    def record(self, name: str, **details: Any) -> None:
        self.steps.append(PipelineStep(name=name, details=details))

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(step) for step in self.steps]
