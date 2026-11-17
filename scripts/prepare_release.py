"""Scan the release files and create an anonymous source archive, without Git history."""

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
    ".env.example",
    "requirements-tested.txt",
}
SOURCE_DIRS = {"src", "configs", "examples", "docs", "scripts", "tests"}
PATTERNS = {
    "API credential": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    "personal absolute path": re.compile(
        r"/(?:home|Users|volume|map-vepfs|mnt|data/users)/[A-Za-z0-9_.-]+/"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "embedded URL credentials": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def release_files():
    candidates = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for name in SOURCE_DIRS:
        if (ROOT / name).is_dir():
            candidates.extend((ROOT / name).rglob("*"))
    for path in sorted(candidates):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(
            part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
            or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            continue
        if (len(relative.parts) == 1 and relative.name in ROOT_FILES) or relative.parts[
            0
        ] in SOURCE_DIRS:
            if path.suffix not in {".pyc", ".pdf"} and path.name != ".env":
                yield path


def scan(files, denied_names=()):
    findings = []
    for path in files:
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            findings.append((str(path.relative_to(ROOT)), "unexpected binary file"))
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append((str(path.relative_to(ROOT)), label))
        for name in denied_names:
            if name.lower() in text.lower():
                findings.append((str(path.relative_to(ROOT)), "denied identifier"))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--deny-name",
        action="append",
        default=[],
        help="Additional private identifier; repeat as needed",
    )
    parser.add_argument("--output", default="dist/WorldMem-anonymous.zip")
    args = parser.parse_args()
    files = list(release_files())
    findings = scan(files, args.deny_name)
    if findings:
        for path, reason in findings:
            print(f"{path}: {reason}")
        raise SystemExit("Release scan failed. Values are intentionally not printed.")
    if not args.check_only:
        target = Path(args.output)
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                item = zipfile.ZipInfo("WorldMem/" + path.relative_to(ROOT).as_posix())
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = (0o755 if path.suffix == ".sh" else 0o644) << 16
                archive.writestr(item, path.read_bytes())
        print(
            f"Prepared {target.name}: {len(files)} source files; no datasets, outputs, credentials, or Git history"
        )
    else:
        print(f"Release scan passed: {len(files)} source files")


if __name__ == "__main__":
    main()
