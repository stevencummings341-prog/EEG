"""Phase 2c embedding replay with safe path resolution.

Never trust stale ``npz_path`` columns from migrated WBCIC indexes. Always
re-resolve via ``embeddings_dir / model / seedN / {subj}_{src}-to-{tgt}.npz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from code.tta.exceptions import FeatureSourceError
from code.tta.feature_sources.base import FeatureBundle, FeatureSource, make_cell_id

PathLike = Union[str, Path]


def resolve_embedding_npz_path(
    *,
    embeddings_dir: PathLike,
    model: str,
    seed: int,
    subject: str,
    source_session: str,
    target_session: str,
    original_path: Optional[PathLike] = None,
    project_root: Optional[PathLike] = None,
) -> Path:
    """Re-compose npz path; fail-fast with a clear diagnosis if missing."""
    emb_dir = Path(embeddings_dir)
    if project_root is not None and not emb_dir.is_absolute():
        emb_dir = Path(project_root) / emb_dir
    resolved = (
        emb_dir
        / model
        / f"seed{int(seed)}"
        / f"{subject}_{source_session}-to-{target_session}.npz"
    )
    if resolved.is_file():
        return resolved.resolve()

    tried = [str(resolved)]
    # Optional: if original exists (e.g. SHU index already correct), accept it.
    if original_path:
        op = Path(original_path)
        if project_root is not None and not op.is_absolute():
            op = Path(project_root) / op
        tried.append(str(op))
        if op.is_file():
            return op.resolve()

    raise FeatureSourceError(
        "embedding npz not found.\n"
        f"  dataset/model/seed/subject/sessions may be wrong or path stale.\n"
        f"  model={model} seed={seed} subject={subject} "
        f"{source_session}->{target_session}\n"
        f"  original_path={original_path}\n"
        f"  tried_paths={tried}"
    )


def _get_arr(npz: Any, key: str) -> Optional[np.ndarray]:
    if key not in npz.files:
        return None
    return np.asarray(npz[key])


class EmbeddingReplaySource(FeatureSource):
    name = "embedding_replay"

    def __init__(
        self,
        *,
        embeddings_dir: PathLike,
        dataset: str,
        project_root: Optional[PathLike] = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.dataset = str(dataset)
        self.project_root = Path(project_root) if project_root else None

    def load_cell(
        self,
        *,
        model: str,
        seed: int,
        subject: str,
        source_session: str,
        target_session: str,
        original_npz_path: Optional[PathLike] = None,
        **_: Any,
    ) -> FeatureBundle:
        resolved = resolve_embedding_npz_path(
            embeddings_dir=self.embeddings_dir,
            model=model,
            seed=seed,
            subject=subject,
            source_session=source_session,
            target_session=target_session,
            original_path=original_npz_path,
            project_root=self.project_root,
        )
        npz = np.load(resolved)
        try:
            src_z = _get_arr(npz, "source_train__z")
            src_y = _get_arr(npz, "source_train__y")
            tgt_z = _get_arr(npz, "target_test__z")
            tgt_y = _get_arr(npz, "target_test__y")
            tgt_logits = _get_arr(npz, "target_test__logits")
            tgt_probs = _get_arr(npz, "target_test__probs")
            tgt_pred = _get_arr(npz, "target_test__pred")
            tgt_conf = _get_arr(npz, "target_test__conf")
        finally:
            npz.close()

        if tgt_z is None:
            raise FeatureSourceError(
                f"npz missing target_test__z: {resolved}"
            )

        cell_id = make_cell_id(
            self.dataset, model, seed, subject, source_session, target_session
        )
        emb_dim = int(tgt_z.shape[1]) if tgt_z.ndim == 2 else None
        return FeatureBundle(
            cell_id=cell_id,
            dataset=self.dataset,
            model=model,
            seed=int(seed),
            subject=subject,
            source_session=source_session,
            target_session=target_session,
            source_features=src_z.astype(np.float32) if src_z is not None else None,
            source_labels=src_y.astype(np.int64) if src_y is not None else None,
            target_features=tgt_z.astype(np.float32),
            target_logits=tgt_logits.astype(np.float32) if tgt_logits is not None else None,
            target_probs=tgt_probs.astype(np.float32) if tgt_probs is not None else None,
            target_pred=tgt_pred.astype(np.int64) if tgt_pred is not None else None,
            target_conf=tgt_conf.astype(np.float32) if tgt_conf is not None else None,
            target_y_true=tgt_y.astype(np.int64) if tgt_y is not None else None,
            feature_source=self.name,
            npz_path_resolved=str(resolved),
            npz_path_original=str(original_npz_path or ""),
            embedding_dim=emb_dim,
            n_source=int(len(src_y)) if src_y is not None else 0,
            n_target=int(len(tgt_z)),
            metadata={"keys_loaded": True},
        )
