"""Smoke / framework reporters — factual only, no scientific claims."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from code.tta.oracle.base import PROVISIONAL_ORACLE_NOTES


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_framework_smoke_report(
    path: Path,
    *,
    status: str,
    notes: Sequence[str],
) -> None:
    lines = [
        "---",
        'title: "TTA Framework Smoke Report"',
        f'created: "{datetime.now().date()}"',
        'status: "scaffold"',
        "---",
        "",
        "# TTA Framework Smoke Report",
        "",
        f"**Status:** {status}",
        "",
        "This report documents that the **TTA backend scaffold is runnable**.",
        "It does **not** claim that T3A is effective or that cross-session drop is fixed.",
        "",
        "## Notes",
        "",
    ]
    for n in notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "## Known limitations",
        "",
        "- Round-1 implements no_tta + one minimal T3A variant only.",
        "- Pretrained model is not integrated yet.",
        "- Method catalog candidates are registered in docs/config, not all implemented.",
        "",
        "## Next integration steps",
        "",
        "- Add a ModelAdapter + config for the senior pretrained model.",
        "- Revisit Oracle thresholds after pretrained integration "
        f"({PROVISIONAL_ORACLE_NOTES})",
        "",
    ]
    _write(path, "\n".join(lines))


def write_replay_validation_report(
    path: Path,
    *,
    n_cells: int,
    n_pass: int,
    n_fail: int,
    max_abs_delta: Optional[float],
    diagnosis: Sequence[str],
) -> None:
    lines = [
        "---",
        'title: "No-TTA Replay Validation Report"',
        f'created: "{datetime.now().date()}"',
        'status: "diagnostic"',
        "---",
        "",
        "# No-TTA Replay Validation",
        "",
        "Replay validation **attempted**. Ideal target: |Δ| < 1e-6 vs Phase 2c "
        "`acc_target` joined by `cell_id`.",
        "",
        f"- cells checked: {n_cells}",
        f"- passed (|Δ|<1e-6): {n_pass}",
        f"- failed / missing: {n_fail}",
        f"- max |Δ|: {max_abs_delta}",
        "",
        "## Diagnosis",
        "",
    ]
    for d in diagnosis:
        lines.append(f"- {d}")
    lines += [
        "",
        "If alignment failed, do **not** expand T3A until path/label/join issues "
        "are resolved.",
        "",
    ]
    _write(path, "\n".join(lines))


def write_t3a_smoke_report(
    path: Path,
    *,
    status: str,
    selected_subjects: Sequence[str],
    n_rows: int,
    extra_notes: Sequence[str] = (),
) -> None:
    lines = [
        "---",
        'title: "Minimal T3A Smoke Report"',
        f'created: "{datetime.now().date()}"',
        'status: "smoke"',
        "---",
        "",
        "# Minimal T3A Smoke Report",
        "",
        f"**Pipeline status:** {status}",
        "",
        "This smoke only checks that the **pipeline can run** "
        "(no_tta + one minimal T3A variant).",
        "**Do not** interpret accuracy deltas as scientific evidence that T3A works "
        "or fails.",
        "",
        f"- selected subjects: {', '.join(selected_subjects) or '(none)'}",
        f"- result rows: {n_rows}",
        "",
        "## Notes",
        "",
    ]
    for n in extra_notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "## Known limitations",
        "",
        "- Single minimal T3A variant; not a full ablation.",
        "- EEGNet appears only as an example adapter name in smoke config.",
        "- Pretrained model not integrated.",
        "",
    ]
    _write(path, "\n".join(lines))


def write_oracle_diagnostic_report(
    path: Path,
    *,
    status: str,
    n_rows: int,
    notes: Sequence[str] = (),
) -> None:
    lines = [
        "---",
        'title: "Oracle Diagnostic Report (minimal)"',
        f'created: "{datetime.now().date()}"',
        'status: "diagnostic_only"',
        "---",
        "",
        "# Oracle Diagnostic Report (minimal)",
        "",
        f"**Status:** {status}",
        "",
        "Oracle methods use target true labels and are **diagnostic only** "
        "(`used_target_labels=True`, `oracle_diagnostic_only=True`, "
        "`not_deployable=True`). They are **not** deployable TTA methods.",
        "",
        f"- result rows: {n_rows}",
        "",
        f"> {PROVISIONAL_ORACLE_NOTES}",
        "",
        "## Notes",
        "",
    ]
    for n in notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "Complex Oracle candidates (shrinkage / reliability / mahalanobis / …) "
        "are catalog-only in Round-1.",
        "",
    ]
    _write(path, "\n".join(lines))


def save_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_full_a0_replay_report(
    path: Path,
    *,
    verdict: str,
    dataset: str,
    universe: Dict[str, Any],
    replay: Dict[str, Any],
    notes: Sequence[str] = (),
) -> None:
    """Full-universe (all canonical Phase 2c cells) no_tta replay validation report.

    Factual only — this documents whether offline embedding replay reproduces
    Phase 2c ``acc_target`` exactly (|Δ| < tolerance). It does not draw any
    scientific conclusion about T3A/Tent/SHOT/Oracle.
    """
    lines = [
        "---",
        'title: "Full A0 No-TTA Replay Validation Report"',
        f'created: "{datetime.now().date()}"',
        'status: "diagnostic"',
        "---",
        "",
        "# Full A0 No-TTA Replay Validation",
        "",
        f"**dataset:** {dataset}",
        f"**verdict:** {verdict}",
        "",
        "Validates that offline embedding replay of **all canonical Phase 2c cells** "
        "reproduces `acc_target` exactly under `no_tta` (frozen-model predictions "
        "already stored in the npz). This is a plumbing/consistency check, "
        "**not** a T3A/TTA effectiveness claim.",
        "",
        "## Canonical universe",
        "",
        f"- metrics cells (label_based + euclidean, requested models/seeds): {universe.get('n_metrics_cells')}",
        f"- index cells (embed_index__*.csv): {universe.get('n_index_cells')}",
        f"- embedding files (npz) found: {universe.get('n_embedding_cells')}",
        f"- canonical valid cells (metrics ∩ embeddings): {universe.get('n_canonical_valid')}",
        f"- metrics duplicate cell_ids: {len(universe.get('metrics_duplicates', []))}",
        f"- index duplicate cell_ids (dup split rows): {len(universe.get('index_duplicate_cells', []))}",
        f"- index cells with incomplete/unexpected split set: {len(universe.get('incomplete_split_cells', []))}",
        f"- embedding duplicate cell_ids: {len(universe.get('embedding_duplicate_cells', []))}",
        f"- missing embeddings (in metrics, no npz found): {len(universe.get('missing_embedding', []))}",
        f"- unexpected embeddings (npz found, not in metrics): {len(universe.get('unexpected_embedding', []))}",
        f"- missing from index (in metrics, absent from embed_index): {len(universe.get('missing_from_index', []))}",
        f"- unexpected in index (in embed_index, absent from metrics): {len(universe.get('unexpected_in_index', []))}",
        "",
        "## Replay results",
        "",
        f"- cells attempted: {replay.get('n_attempted')}",
        f"- cells load/eval failed: {replay.get('n_failed')}",
        f"- cells replayed successfully: {replay.get('n_ok')}",
        f"- passed \\|Δ\\| < {replay.get('tolerance')}: {replay.get('n_pass')}",
        f"- failed tolerance: {replay.get('n_fail_tolerance')}",
        f"- max \\|Δ\\|: {replay.get('max_abs_delta')}",
        f"- wall time (s): {replay.get('elapsed_sec')}",
        "",
    ]
    by = replay.get("by_model_seed_direction")
    if by:
        lines += ["## Max |Δ| by model / seed / direction (worst 20)", ""]
        lines += ["| model | seed | direction | n | max_abs_delta |", "|---|---|---|---|---|"]
        for row in by[:20]:
            lines.append(
                f"| {row.get('model')} | {row.get('seed')} | {row.get('direction')} "
                f"| {row.get('n')} | {row.get('max_abs_delta')} |"
            )
        lines.append("")
    lines += ["## Notes", ""]
    for n in notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "If verdict is `partial` or `blocked`, see the accompanying "
        "`full_a0_universe_consistency.csv` and `full_a0_replay_delta_table.csv` "
        "for the exact mismatch/failure rows. Do **not** expand T3A/Oracle "
        "matrices until this is `complete`.",
        "",
    ]
    _write(path, "\n".join(lines))
