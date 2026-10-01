"""One JSONL loader for every eval script (eval_analysis, paired_system_comparison,
judge_attribution, ensemble_posthoc).

Rules (the same everywhere, so two scripts can never score different row sets):
  * blank lines are ignored; unparsable lines are skipped and counted (warning);
  * resumed runs append, so an image_path can appear more than once: per
    image_path the LAST row with ``error is None`` wins;
  * an image_path that only ever produced error rows is kept as its last error
    row, flagged ``_error_only=True``; scorers count it as WRONG (never dropped)
    and report how many there were;
  * a row without an image_path is kept as-is (keyed by line number).
Output order is the order in which each image_path first appeared.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ERROR_FLAG = "_error_only"


@dataclass
class LoadStats:
    path: str
    n_lines: int = 0
    n_unparsable: int = 0
    n_rows: int = 0
    n_unique: int = 0
    n_duplicate_rows: int = 0        # rows superseded by a later row for the same image
    n_error_only: int = 0            # image_paths with no successful row
    error_only_paths: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{Path(self.path).name}: {self.n_unique} images "
                f"({self.n_rows} rows, {self.n_duplicate_rows} superseded duplicates, "
                f"{self.n_unparsable} unparsable lines, {self.n_error_only} error-only images)")


def is_error_row(rec: dict) -> bool:
    return rec.get("error") is not None


def load_records(path: str | Path, verbose: bool = True) -> tuple[list[dict], LoadStats]:
    """Load and de-duplicate an eval JSONL. Returns (records, stats)."""
    stats = LoadStats(path=str(path))
    order: list[str] = []
    last_ok: dict[str, dict] = {}
    last_err: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            stats.n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.n_unparsable += 1
                continue
            if not isinstance(rec, dict):
                stats.n_unparsable += 1
                continue
            stats.n_rows += 1
            key = rec.get("image_path") or f"__line{lineno}"
            if key not in last_ok and key not in last_err:
                order.append(key)
            if is_error_row(rec):
                last_err[key] = rec
            else:
                last_ok[key] = rec

    records = []
    for key in order:
        if key in last_ok:
            records.append(last_ok[key])
        else:
            rec = dict(last_err[key])
            rec[ERROR_FLAG] = True
            records.append(rec)
            stats.error_only_paths.append(key)
    stats.n_unique = len(records)
    stats.n_error_only = len(stats.error_only_paths)
    stats.n_duplicate_rows = stats.n_rows - stats.n_unique

    if verbose and (stats.n_unparsable or stats.n_error_only or stats.n_duplicate_rows):
        print(f"[jsonl_io] {stats.summary()}", file=sys.stderr)
        if stats.n_unparsable:
            print(f"[jsonl_io] WARNING: skipped {stats.n_unparsable} unparsable line(s) "
                  f"in {path}", file=sys.stderr)
    return records, stats


def by_image_path(records: list[dict]) -> dict[str, dict]:
    return {r.get("image_path"): r for r in records if r.get("image_path")}


def load_df(path: str | Path, verbose: bool = True):
    """Same as load_records, as a DataFrame with the LoadStats in ``df.attrs['load_stats']``."""
    import pandas as pd

    records, stats = load_records(path, verbose=verbose)
    df = pd.DataFrame(records)
    if ERROR_FLAG not in df.columns:
        df[ERROR_FLAG] = False
    df[ERROR_FLAG] = df[ERROR_FLAG].fillna(False).astype(bool)
    df.attrs["load_stats"] = stats
    return df, stats
