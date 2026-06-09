"""Standardized experiment result helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ExperimentResult:
    name: str
    objective: Optional[float] = None
    feasible: Optional[bool] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_result(result: ExperimentResult, output_dir, config=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_path = output_path / 'result.json'
    result_payload = _json_safe(result.to_dict())
    result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True))

    if config is not None:
        config_path = output_path / 'config.json'
        config_path.write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True))

    return result_path


def load_result(output_dir):
    output_path = Path(output_dir)
    payload = json.loads((output_path / 'result.json').read_text())
    return ExperimentResult(**payload)
