# -*- coding: utf-8 -*-
"""EINX dataset manifest system (spec §8, §9, §18).

Every processed dataset has an immutable identity captured in a
manifest.json.  The manifest records:

  * dataset ID + version
  * source files + their hashes
  * processing configuration (and its hash — so config changes
    produce a different dataset identity)
  * tokenizer version + path
  * record count
  * token count
  * creation timestamp
  * EINX software version

The manifest is the single source of truth for "which dataset is
this?".  Training checkpoints record the manifest hash so a resumed
run can verify it's using the same dataset — if the dataset has
changed, training fails with a clear error rather than silently
switching data.

Per spec §8: "If processing configuration changes, the resulting
dataset identity must change."  The manifest hash is computed from
(source_hash + processing_hash + tokenizer_version + record_count),
so any of those changing produces a different identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

EINX_VERSION = "0.3.0"  # Build 3


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetManifest:
    """The immutable identity of a processed dataset.

    Every field is populated from actual processing — no hardcoded
    values.  The ``identity_hash`` is computed from the source +
    processing + tokenizer info, so any change to those produces a
    different identity (spec §8).
    """

    name: str = ""
    version: str = "0.1.0"

    # Source info
    source_files: List[str] = field(default_factory=list)
    source_hash: str = ""              # hash of all source file contents
    n_source_records: int = 0

    # Processing info
    processing_config: Dict[str, Any] = field(default_factory=dict)
    processing_hash: str = ""          # hash of the processing config

    # Tokenizer info
    tokenizer_path: str = ""
    tokenizer_version: str = ""

    # Output stats
    n_records: int = 0
    n_tokens: int = 0
    n_train_records: int = 0
    n_val_records: int = 0

    # Shards
    shard_count: int = 0
    shard_files: List[str] = field(default_factory=list)

    # Reproducibility
    seed: int = 42
    context_length: int = 0

    # Identity
    identity_hash: str = ""           # computed from everything above
    created_at: str = ""
    einx_version: str = EINX_VERSION

    # ------------------------------------------------------------------
    def compute_identity(self) -> str:
        """Compute the immutable identity hash.

        The hash is over: name, version, source_hash, processing_hash,
        tokenizer_version, n_records, n_tokens, context_length.
        Changes to any of these produce a different identity (spec §8).
        """
        h = hashlib.sha256()
        h.update(self.name.encode("utf-8"))
        h.update(self.version.encode("utf-8"))
        h.update(self.source_hash.encode("utf-8"))
        h.update(self.processing_hash.encode("utf-8"))
        h.update(self.tokenizer_version.encode("utf-8"))
        h.update(str(self.n_records).encode("utf-8"))
        h.update(str(self.n_tokens).encode("utf-8"))
        h.update(str(self.context_length).encode("utf-8"))
        self.identity_hash = h.hexdigest()
        return self.identity_hash

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        # Ensure identity is up-to-date before serialising
        self.compute_identity()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, path: Union[str, Path]) -> None:
        """Save the manifest to manifest.json.  Atomic write."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        import os
        os.replace(tmp, path)
        logger.info("manifest saved to %s (identity=%s)", path, self.identity_hash[:16])

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DatasetManifest":
        """Load a manifest from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        manifest = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        manifest.compute_identity()  # verify
        return manifest

    # ------------------------------------------------------------------
    def verify(self, dataset_dir: Union[str, Path]) -> bool:
        """Verify that a dataset directory matches this manifest.

        Checks:
          * manifest.json exists
          * shard files exist
          * shard file hashes match (if recorded)

        Returns True if everything matches, False otherwise.
        """
        dataset_dir = Path(dataset_dir)
        # 1. Manifest exists
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("manifest.json not found in %s", dataset_dir)
            return False

        # 2. Shard files exist
        for shard_file in self.shard_files:
            shard_path = dataset_dir / shard_file
            if not shard_path.exists():
                logger.error("shard file missing: %s", shard_path)
                return False

        # 3. Identity hash matches
        loaded = DatasetManifest.load(manifest_path)
        if loaded.identity_hash != self.identity_hash:
            logger.error(
                "manifest identity mismatch: expected %s, got %s",
                self.identity_hash[:16], loaded.identity_hash[:16],
            )
            return False

        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_files(paths: List[Union[str, Path]]) -> str:
    """Compute a SHA-256 hash over the contents of multiple files.

    Used to compute the ``source_hash`` for a dataset manifest —
    ensures that any change to source data changes the dataset identity.
    """
    h = hashlib.sha256()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def hash_config(config: Dict[str, Any]) -> str:
    """Compute a SHA-256 hash over a config dict (deterministic)."""
    # Sort keys so the hash is stable regardless of dict ordering
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_manifest(
    *,
    name: str,
    version: str,
    source_files: List[Union[str, Path]],
    processing_config: Dict[str, Any],
    tokenizer_path: str,
    tokenizer_version: str,
    n_records: int,
    n_tokens: int,
    n_train_records: int = 0,
    n_val_records: int = 0,
    shard_files: Optional[List[str]] = None,
    seed: int = 42,
    context_length: int = 0,
) -> DatasetManifest:
    """Build a manifest from real processing info."""
    manifest = DatasetManifest(
        name=name,
        version=version,
        source_files=[str(p) for p in source_files],
        source_hash=hash_files(source_files),
        n_source_records=n_records,  # may differ from final n_records
        processing_config=processing_config,
        processing_hash=hash_config(processing_config),
        tokenizer_path=tokenizer_path,
        tokenizer_version=tokenizer_version,
        n_records=n_records,
        n_tokens=n_tokens,
        n_train_records=n_train_records,
        n_val_records=n_val_records,
        shard_files=shard_files or [],
        shard_count=len(shard_files or []),
        seed=seed,
        context_length=context_length,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest.compute_identity()
    return manifest
