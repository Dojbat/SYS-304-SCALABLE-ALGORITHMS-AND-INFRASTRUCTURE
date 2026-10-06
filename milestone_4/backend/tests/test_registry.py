import json
from concurrent.futures import ThreadPoolExecutor

from app.registry import (
    MODEL_FILE,
    ensure_seeded,
    read_registry,
    resolve_current,
    write_registry,
)


def _make_seed(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / MODEL_FILE).write_bytes(b"onnx")
    (seed / "vocab.txt").write_text("vocab")
    return seed


def test_seeding_copies_the_seed_in_as_v1(tmp_path):
    seed = _make_seed(tmp_path)
    models = tmp_path / "models"

    assert ensure_seeded(models, seed)

    assert (models / "v1" / MODEL_FILE).read_bytes() == b"onnx"
    registry = read_registry(models)
    assert registry["current"] == "v1"
    assert registry["versions"]["v1"]["adapter_dir"] is None
    assert resolve_current(models, seed).path == models / "v1"


def test_seeding_leaves_an_existing_registry_alone(tmp_path):
    seed = _make_seed(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    write_registry(models, {"current": "v3", "versions": {}})

    assert ensure_seeded(models, seed)

    assert read_registry(models)["current"] == "v3"
    assert not (models / "v1").exists()


def test_without_a_registry_the_seed_is_served_directly(tmp_path):
    seed = _make_seed(tmp_path)
    ref = resolve_current(tmp_path / "missing", seed)
    assert ref.version == "v1"
    assert ref.path == seed


def test_registry_write_is_valid_json_with_no_temp_file_left(tmp_path):
    write_registry(tmp_path, {"current": "v2", "versions": {"v2": {}}})
    assert json.loads((tmp_path / "registry.json").read_text())["current"] == "v2"
    assert [p.name for p in tmp_path.iterdir()] == ["registry.json"]


def test_concurrent_seeding_by_several_replicas_is_safe(tmp_path):
    seed = _make_seed(tmp_path)
    models = tmp_path / "models"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ensure_seeded(models, seed), range(8)))

    assert all(results)
    assert read_registry(models)["current"] == "v1"
    assert (models / "v1" / MODEL_FILE).read_bytes() == b"onnx"
    # no leftover staging dirs or temp files
    assert sorted(p.name for p in models.iterdir()) == ["registry.json", "v1"]


def test_seeding_finishes_when_another_replica_already_copied_v1(tmp_path):
    seed = _make_seed(tmp_path)
    models = tmp_path / "models"
    (models / "v1").mkdir(parents=True)
    (models / "v1" / MODEL_FILE).write_bytes(b"onnx")

    assert ensure_seeded(models, seed)
    assert read_registry(models)["current"] == "v1"
