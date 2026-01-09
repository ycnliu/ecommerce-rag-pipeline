#!/usr/bin/env python3
"""
Generate deployment manifest with metadata and checksums.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def generate_manifest(output_path: str, version: str, source_dir: str = "space"):
    """Generate deployment manifest with file hashes."""

    source_dir = Path(source_dir)

    manifest = {
        "version": version,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": {},
        "metadata": {
            "deployment_type": "huggingface_spaces",
            "environment": "production" if "main" in version else "preview",
        },
    }

    # Calculate hashes for key files
    files_to_track = ["app.py", "requirements.txt", "demo_products.csv"]

    for file_name in files_to_track:
        file_path = source_dir / file_name

        if file_path.exists():
            manifest["files"][file_name] = {
                "hash": calculate_file_hash(file_path),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat(),
            }

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save manifest
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest generated: {output_path}")
    print(f"Version: {version}")
    print(f"Files tracked: {len(manifest['files'])}")


def main():
    parser = argparse.ArgumentParser(description="Generate deployment manifest")
    parser.add_argument("--output", required=True, help="Output manifest file")
    parser.add_argument("--version", required=True, help="Version/commit hash")
    parser.add_argument("--source-dir", default="space", help="Source directory")

    args = parser.parse_args()

    generate_manifest(args.output, args.version, args.source_dir)


if __name__ == "__main__":
    main()
