"""
Stage registry.

Stages register under a dotted name that mirrors their module path, e.g.
`pipeline/stages/inference/slice_extraction/axial_sampler.py` registers
"inference.slice_extraction.axial_sampler" — the host system's convention.

Keep this module (and stage modules at import time) free of torch/model
imports: `discover()` imports every stage module, and that must stay cheap.
Heavy imports belong inside stage constructors or `run()`.
"""

import importlib
import inspect
import pkgutil
from typing import Any, Union

_REGISTRY: dict[str, type] = {}
_discovered = False

StageSpec = Union[str, dict]


def register(name: str):
    """Class decorator: register a Stage subclass under `name`."""

    def decorator(cls: type) -> type:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Stage '{name}' already registered by {existing.__module__}.{existing.__qualname__}"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def discover() -> None:
    """Import every module under pipeline.stages so their @register decorators run."""
    global _discovered
    if _discovered:
        return
    import pipeline.stages as root

    for info in pkgutil.walk_packages(root.__path__, prefix=f"{root.__name__}."):
        importlib.import_module(info.name)
    _discovered = True


def get(name: str) -> type:
    discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown stage '{name}'. Available: {', '.join(available())}") from None


def available() -> list[str]:
    discover()
    return sorted(_REGISTRY)


def parse_spec(spec: StageSpec) -> tuple[str, dict]:
    """`"name"` or `{"name": {kwargs}}` → (name, kwargs)."""
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict) and len(spec) == 1:
        name, kwargs = next(iter(spec.items()))
        return name, dict(kwargs or {})
    raise ValueError(f"Bad stage spec {spec!r}: expected 'name' or {{name: {{kwargs}}}}")


def build(spec: StageSpec, resources: Any = None):
    """Instantiate a stage from its spec.

    `resources` (shared model handles) is passed only to stages whose __init__
    accepts it, so host-system stages like AxialSliceExtractor work unmodified.
    """
    name, kwargs = parse_spec(spec)
    cls = get(name)
    if "resources" in inspect.signature(cls.__init__).parameters:
        kwargs.setdefault("resources", resources)
    return cls(**kwargs)
