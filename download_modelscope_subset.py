from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path


def unmatched_patterns(root: Path, patterns: list[str]) -> list[str]:
    files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    return [
        pattern
        for pattern in patterns
        if not any(fnmatch.fnmatchcase(relative, pattern) for relative in files)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only selected files from a ModelScope dataset repository"
    )
    parser.add_argument("--dataset", default="hy2628982280/WildFake")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--files", nargs="+", required=True, help="Exact paths or glob patterns")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--international",
        action="store_true",
        help="Use the modelscope.ai endpoint; recommended for hosts outside mainland China",
    )
    args = parser.parse_args()

    if args.international:
        os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"

    try:
        from modelscope import dataset_snapshot_download
    except ImportError as exc:
        raise ImportError("Install ModelScope first: python -m pip install -U modelscope") from exc

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    result = dataset_snapshot_download(
        dataset_id=args.dataset,
        local_dir=str(local_dir),
        allow_patterns=args.files,
        max_workers=args.max_workers,
    )
    result_path = Path(result).expanduser().resolve()
    missing = unmatched_patterns(result_path, args.files)
    if missing:
        raise RuntimeError(
            "ModelScope download completed but these requested patterns matched no local files: "
            f"{missing}"
        )
    print(f"Downloaded and verified {len(args.files)} pattern(s) under: {result_path}")


if __name__ == "__main__":
    main()
