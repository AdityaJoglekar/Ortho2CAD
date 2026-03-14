import argparse
from pathlib import Path

import cairosvg


def iter_svg_files(src_dir: Path):
    for path in src_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".svg":
            yield path


def svg_to_png(src_dir: Path, dst_dir: Path, *, overwrite: bool, limit: int | None):
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    converted = 0
    skipped = 0
    failed = 0

    for svg_path in sorted(iter_svg_files(src_dir)):
        total += 1
        if limit is not None and converted + skipped + failed >= limit:
            break

        rel = svg_path.relative_to(src_dir)
        out_dir = dst_dir / rel.parent
        out_path = out_dir / (svg_path.stem + ".png")

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            cairosvg.svg2png(url=str(svg_path), write_to=str(out_path))
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {svg_path} -> {out_path} :: {exc}")

        # Lightweight progress for long runs.
        if (converted + skipped + failed) % 200 == 0:
            print(
                f"progress: seen={total} converted={converted} skipped={skipped} failed={failed}"
            )

    print(f"done: seen={total} converted={converted} skipped={skipped} failed={failed}")


def main():
    parser = argparse.ArgumentParser(description="Convert SVGs to PNGs (recursive).")
    parser.add_argument(
        "--src",
        default="orthographic_out",
        help="Source directory containing .svg files (possibly nested).",
    )
    parser.add_argument(
        "--dst",
        default="orthographic_out_png",
        help="Destination directory where .png files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .png files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most N files (useful for smoke tests).",
    )
    args = parser.parse_args()

    svg_to_png(Path(args.src), Path(args.dst), overwrite=args.overwrite, limit=args.limit)


if __name__ == "__main__":
    main()