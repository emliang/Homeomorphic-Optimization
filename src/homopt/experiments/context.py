"""Shared runtime context for experiment execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class ExperimentContext:
    """Execution context propagated across config/script entrypoints."""

    entrypoint: str
    name: Optional[str] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    output_dir: Optional[str] = None
    preset: Optional[str] = None
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        output_dir,
        *,
        entrypoint: str = "config",
        preset: Optional[str] = None,
    ):
        runtime = dict(config.get("runtime", {}))
        return cls(
            entrypoint=entrypoint,
            name=config.get("name"),
            seed=runtime.get("seed"),
            device=runtime.get("device"),
            dtype=runtime.get("dtype"),
            output_dir=str(Path(output_dir)),
            preset=preset,
            config_path=config.get("_config_path"),
        )

    @classmethod
    def from_script(cls, name: str, params: Mapping[str, Any], output_dir):
        params = dict(params or {})
        return cls(
            entrypoint="script",
            name=name,
            seed=params.get("seed"),
            device=params.get("device"),
            dtype=params.get("dtype"),
            output_dir=str(Path(output_dir)),
        )
