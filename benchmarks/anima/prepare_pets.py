"""Prepare a small, deterministic Oxford-IIIT Pet diffusion benchmark subset.

The archive is intentionally consumed as a stream: only selected, validated
JPEG/PNG bytes are written to the output directory.  The full archive must
still be downloaded when using ``--download`` (the upstream archive is about
800 MB).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised by the CLI
    raise SystemExit("Pillow is required to prepare the dataset") from exc


SOURCE_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
LICENSE = "CC BY-SA 4.0"
CLASSES = ("Bengal", "Siamese", "beagle", "shiba_inu")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Candidate:
    name: str
    class_name: str
    data: bytes
    sha256: str
    dhash: str
    rank: str


def _safe_member_name(name: str) -> str | None:
    """Return a normalized relative name, rejecting unsafe tar entries."""
    name = name.replace("\\", "/")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or name.startswith("/"):
        return None
    return "/".join(parts)


def _class_for_name(name: str) -> str | None:
    stem = Path(name).stem
    for class_name in CLASSES:
        if stem.lower().startswith(class_name.lower() + "_"):
            return class_name
    return None


def _dhash(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("L").resize((9, 8))
        pixels = list(image.getdata())
    bits = [pixels[row * 9 + col] > pixels[row * 9 + col + 1] for row in range(8) for col in range(8)]
    return "".join("1" if bit else "0" for bit in bits)


def _distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def _rank(seed: int, digest: str, name: str) -> str:
    return hashlib.sha256(f"{seed}:{digest}:{name}".encode()).hexdigest()


def _caption(class_name: str) -> str:
    animal = "cat" if class_name in {"Bengal", "Siamese"} else "dog"
    return f"a photo of a {class_name.replace('_', ' ')} {animal}"


def prepare(archive: Path, output: Path, seed: int = 0, source_url: str = SOURCE_URL) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    candidates = {class_name: [] for class_name in CLASSES}
    source_hash = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            source_hash.update(chunk)
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            safe_name = _safe_member_name(member.name)
            if safe_name is None or not member.isfile():
                continue
            class_name = _class_for_name(safe_name)
            if class_name is None or Path(safe_name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            data = handle.read()
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                digest = hashlib.sha256(data).hexdigest()
                image_hash = _dhash(data)
            except Exception:
                continue
            candidates[class_name].append(Candidate(safe_name, class_name, data, digest, image_hash, _rank(seed, digest, safe_name)))

    selected = {class_name: [] for class_name in CLASSES}
    seen_sha256: set[str] = set()
    seen_dhash: list[str] = []
    for item in sorted((item for items in candidates.values() for item in items), key=lambda candidate: candidate.rank):
        if len(selected[item.class_name]) >= 40 or item.sha256 in seen_sha256:
            continue
        if any(_distance(existing, item.dhash) <= 2 for existing in seen_dhash):
            continue
        selected[item.class_name].append(item)
        seen_sha256.add(item.sha256)
        seen_dhash.append(item.dhash)

    if any(len(items) < 40 for items in selected.values()):
        counts = ", ".join(f"{name}={len(items)}" for name, items in selected.items())
        raise ValueError(f"need 40 unique valid images per class; found {counts}")

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for class_name in CLASSES:
        for index, item in enumerate(selected[class_name]):
            split = "train" if index < 24 else "val" if index < 32 else "test"
            filename = f"{class_name}_{index:03d}{Path(item.name).suffix.lower()}"
            destination = output / split / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item.data)
            caption = _caption(class_name)
            (destination.with_suffix(".txt")).write_text(caption, encoding="utf-8")
            records.append({
                "path": destination.relative_to(output).as_posix(),
                "caption": caption,
                "class": class_name,
                "split": split,
                "source_member": item.name,
                "source_sha256": item.sha256,
                "sha256": hashlib.sha256(item.data).hexdigest(),
                "dhash": item.dhash,
                "license": LICENSE,
                "source_url": source_url,
            })
    manifest = {"seed": seed, "source_archive_sha256": source_hash.hexdigest(), "license": LICENSE, "source_url": source_url, "records": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", help="local images.tar.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true", help="download the complete upstream archive")
    parser.add_argument("--url", default=SOURCE_URL)
    args = parser.parse_args()
    if args.download:
        if args.archive is None:
            args.archive = args.output.parent / "images.tar.gz"
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(args.url, timeout=60) as response, args.archive.open("wb") as target:
            shutil.copyfileobj(response, target)
    if args.archive is None:
        parser.error("provide archive or use --download")
    prepare(args.archive, args.output, args.seed, args.url)


if __name__ == "__main__":
    main()
