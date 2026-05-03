import hashlib
import sys
from pathlib import Path

CHUNK = 1 << 20


def compute_dir_sha(root: Path) -> dict:
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    total_bytes = 0
    for p in files:
        rel = p.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        size = p.stat().st_size
        h.update(size.to_bytes(8, "big"))
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
        total_bytes += size
    return {
        "sha256": h.hexdigest(),
        "file_count": len(files),
        "size_bytes": total_bytes,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python compute_dataset_sha.py <dataset_dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    info = compute_dir_sha(root)
    print(f"# 아래 블록을 gogo_ai_backend/models.yaml의 profanity_filter.train_dataset에 paste")
    print(f"  train_dataset:")
    print(f"    uri: {root}")
    print(f"    sha256: {info['sha256']}")
    print(f"    file_count: {info['file_count']}")
    print(f"    size_bytes: {info['size_bytes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
