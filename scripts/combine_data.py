#!/usr/bin/env python3

from pathlib import Path
import shutil
import sys


DATASETS = ("front", "scan", "side")
SUBDIRS = ("images", "labels")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "data"

    missing_sources = [name for name in DATASETS if not (data_root / name).exists()]
    if missing_sources:
        print(f"Missing dataset directories: {', '.join(missing_sources)}", file=sys.stderr)
        return 1

    dest_root = data_root / "all"
    if dest_root.exists():
        shutil.rmtree(dest_root)
    for subdir in SUBDIRS:
        (dest_root / subdir).mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for subdir in SUBDIRS:
            source_dir = data_root / dataset / subdir
            if not source_dir.exists():
                print(f"Skipping missing directory: {source_dir}", file=sys.stderr)
                continue

            for item in source_dir.iterdir():
                if item.name.startswith(".") or not item.is_file():
                    continue

                destination = dest_root / subdir / item.name
                if destination.exists():
                    raise FileExistsError(
                        f"File collision detected for {destination.name} in {destination.parent}"
                    )
                shutil.copy2(item, destination)

    print(f"Combined datasets {', '.join(DATASETS)} into {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

