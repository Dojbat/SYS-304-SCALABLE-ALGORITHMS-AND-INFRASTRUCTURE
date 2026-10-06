"""Model registry on a shared folder.

The backend and the retraining pipeline share one directory (a bind mount,
`milestone_4/models/` on the host, `/models` in the container):

    models/
      registry.json      {"current": "v2", "versions": {"v1": {...}, "v2": {...}}}
      v1/                model_int8.onnx + tokenizer files
      v2/                model_int8.onnx + tokenizer files + adapter/

The retraining pipeline writes a new vN/ directory, then atomically rewrites
registry.json to point `current` at it. The backend only ever reads.

The image also carries the original model (exported at build time) in a
seed directory. On first start, if the shared folder has no registry yet,
the backend copies the seed in as v1. If the shared folder isn't writable
(e.g. a bind mount owned by another uid in CI), it serves the seed directly.
"""

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("disaster-tweet-api")

REGISTRY_FILE = "registry.json"
MODEL_FILE = "model_int8.onnx"
SEED_VERSION = "v1"


@dataclass(frozen=True)
class ModelRef:
    version: str
    path: Path


def read_registry(models_dir: Path) -> dict | None:
    registry_path = models_dir / REGISTRY_FILE
    if not registry_path.is_file():
        return None
    return json.loads(registry_path.read_text(encoding="utf-8"))


def write_registry(models_dir: Path, registry: dict) -> None:
    """Write to a temp file and rename over the old one, so a reader never
    sees a half-written registry.json."""
    # unique temp name: several replicas may seed at the same moment
    tmp_path = models_dir / f"{REGISTRY_FILE}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    try:
        os.replace(tmp_path, models_dir / REGISTRY_FILE)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def _registry_readable(models_dir: Path, attempts: int = 10) -> bool:
    """Windows briefly locks a file while another process replaces it, so a
    read racing a replica's write is retried rather than taken as missing."""
    for _ in range(attempts):
        try:
            return read_registry(models_dir) is not None
        except (OSError, ValueError):
            time.sleep(0.02)
    return False


def ensure_seeded(models_dir: Path, seed_dir: Path) -> bool:
    """Copy the seed model in as v1 if the shared folder has no registry.
    Returns False if the folder isn't usable (missing seed, not writable)."""
    if _registry_readable(models_dir, attempts=1):
        return True
    if not (seed_dir / MODEL_FILE).is_file():
        return False
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        # Copy into a private temp dir, then rename it into place: with several
        # replicas starting together, nobody ever reads a half-copied v1.
        target = models_dir / SEED_VERSION
        if not target.exists():
            staging = models_dir / f".seed-{uuid.uuid4().hex}"
            shutil.copytree(seed_dir, staging)
            try:
                os.rename(staging, target)
            except OSError:  # another replica got there first
                shutil.rmtree(staging, ignore_errors=True)
        if _registry_readable(models_dir):
            return True
        write_registry(
            models_dir,
            {
                "current": SEED_VERSION,
                "versions": {
                    SEED_VERSION: {
                        "created_at": datetime.now(UTC).isoformat(),
                        "source": "seed",
                        # None means "the original LoRA checkpoint in training/"
                        "adapter_dir": None,
                    }
                },
            },
        )
    except OSError:
        # e.g. a replica racing us to the registry rename (Windows refuses
        # concurrent replaces); fine as long as one of us succeeded
        if _registry_readable(models_dir):
            return True
        logger.warning("could not seed %s, serving the seed model directly", models_dir)
        return False
    logger.info("seeded %s with %s", models_dir, SEED_VERSION)
    return True


def resolve_current(models_dir: Path, seed_dir: Path) -> ModelRef:
    registry = read_registry(models_dir)
    if registry is not None:
        version = registry["current"]
        return ModelRef(version=version, path=models_dir / version)
    return ModelRef(version=SEED_VERSION, path=seed_dir)
