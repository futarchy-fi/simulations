"""Discovery and loading for agent/mechanism submissions."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from .errors import DuplicateSubmissionError, InvalidSubmissionError


@dataclass
class SubmissionRegistry:
    """Collection of discovered submission classes."""

    agents: dict[str, type[Any]]
    mechanisms: dict[str, type[Any]]

    def create_agent(self, agent_id: str, params: dict[str, Any]) -> Any:
        try:
            cls = self.agents[agent_id]
        except KeyError as exc:
            raise InvalidSubmissionError(f"Unknown agent id: {agent_id}") from exc
        return cls(**params)

    def create_mechanism(self, mechanism_id: str, params: dict[str, Any]) -> Any:
        try:
            cls = self.mechanisms[mechanism_id]
        except KeyError as exc:
            raise InvalidSubmissionError(f"Unknown mechanism id: {mechanism_id}") from exc
        return cls(**params)


def discover_submissions(
    repo_dirs: Sequence[str | Path],
    extension_dirs: Sequence[str | Path] | None = None,
) -> SubmissionRegistry:
    """Auto-discover agent and mechanism classes from submission folders."""

    extension_dirs = extension_dirs or []
    roots = [Path(p).resolve() for p in [*repo_dirs, *extension_dirs]]

    agents: dict[str, type[Any]] = {}
    mechanisms: dict[str, type[Any]] = {}

    for root in roots:
        _scan_kind(root, "agents", agents)
        _scan_kind(root, "mechanisms", mechanisms)

    return SubmissionRegistry(agents=agents, mechanisms=mechanisms)


def _scan_kind(root: Path, kind: str, registry: dict[str, type[Any]]) -> None:
    folder = root / kind
    if not folder.exists():
        return

    for file_path in sorted(folder.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        module = _load_module(file_path)
        classes = _extract_classes(module, kind)
        if not classes:
            raise InvalidSubmissionError(f"No valid {kind[:-1]} class found in {file_path}")
        for cls in classes:
            if kind == "agents":
                submission_id = getattr(cls, "agent_id")
            else:
                submission_id = getattr(cls, "mechanism_id")

            if submission_id in registry:
                previous = registry[submission_id]
                raise DuplicateSubmissionError(
                    f"Duplicate {kind[:-1]} id '{submission_id}' in {file_path} "
                    f"(already defined by {previous.__module__}.{previous.__name__})"
                )
            registry[submission_id] = cls


def _load_module(file_path: Path) -> ModuleType:
    digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()
    module_name = f"proposal_poker_submission_{digest}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise InvalidSubmissionError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise InvalidSubmissionError(f"Failed importing {file_path}: {exc}") from exc
    return module


def _extract_classes(module: ModuleType, kind: str) -> list[type[Any]]:
    classes: list[type[Any]] = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__:
            continue
        if kind == "agents" and _is_valid_agent_class(cls):
            classes.append(cls)
        if kind == "mechanisms" and _is_valid_mechanism_class(cls):
            classes.append(cls)
    return classes


def _is_valid_agent_class(cls: type[Any]) -> bool:
    submission_id = getattr(cls, "agent_id", None)
    if not isinstance(submission_id, str) or not submission_id:
        return False
    return callable(getattr(cls, "act", None))


def _is_valid_mechanism_class(cls: type[Any]) -> bool:
    submission_id = getattr(cls, "mechanism_id", None)
    if not isinstance(submission_id, str) or not submission_id:
        return False

    required = [
        "init",
        "publish",
        "on_contribution",
        "on_round_end",
        "outcome",
        "valid_data",
    ]
    return all(callable(getattr(cls, name, None)) for name in required)
