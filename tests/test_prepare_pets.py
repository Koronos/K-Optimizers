import io
import json
import random
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from benchmarks.anima.prepare_pets import prepare  # noqa: E402
from PIL import Image  # noqa: E402


def _build_archive(path: Path, *, reverse: bool = False, unsafe: bool = False) -> None:
    entries = []
    for class_name in ("Bengal", "Siamese", "beagle", "shiba_inu"):
        for index in range(40):
            rng = random.Random(f"{class_name}:{index}")
            image = Image.new("RGB", (16, 16))
            image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(16 * 16)])
            stream = io.BytesIO()
            image.save(stream, format="JPEG")
            entries.append((f"images/{class_name}_{index + 1}.jpg", stream.getvalue()))
    if reverse:
        entries.reverse()
    with tarfile.open(path, "w:gz") as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if unsafe:
            info = tarfile.TarInfo("../outside.jpg")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"bad!"))


def _selection(manifest: dict) -> list[dict]:
    keys = ("path", "source_member", "sha256", "split", "class")
    return [{key: record[key] for key in keys} for record in manifest["records"]]


def test_prepare_pets_is_order_independent_and_rejects_unsafe_member(tmp_path):
    first_archive = tmp_path / "first.tar.gz"
    reverse_archive = tmp_path / "reverse.tar.gz"
    _build_archive(first_archive, unsafe=True)
    _build_archive(reverse_archive, reverse=True)
    first = prepare(first_archive, tmp_path / "first-out", seed=7)
    reverse = prepare(reverse_archive, tmp_path / "reverse-out", seed=7)
    assert _selection(first) == _selection(reverse)


def test_prepare_pets_rejects_nonempty_output(tmp_path):
    archive = tmp_path / "images.tar.gz"
    _build_archive(archive)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output must be empty"):
        prepare(archive, output)


def test_prepare_pets_writes_expected_splits(tmp_path):
    archive = tmp_path / "images.tar.gz"
    _build_archive(archive)
    output = tmp_path / "out"
    manifest = prepare(archive, output, seed=7)
    assert len(manifest["records"]) == 160
    assert sum(record["split"] == "train" for record in manifest["records"]) == 96
    assert sum(record["split"] == "val" for record in manifest["records"]) == 32
    assert sum(record["split"] == "test" for record in manifest["records"]) == 32
    assert all((output / record["path"]).is_file() for record in manifest["records"])
    assert json.loads((output / "manifest.json").read_text())["seed"] == 7
