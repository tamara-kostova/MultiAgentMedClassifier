# Lazy (PEP 562): `import pipeline.registry` / `pipeline.context` must not pull in
# torch + LangGraph + all agents just because the package __init__ runs.
__all__ = ["build_pipeline"]


def __getattr__(name):
    if name == "build_pipeline":
        from pipeline.graph import build_pipeline

        return build_pipeline
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
