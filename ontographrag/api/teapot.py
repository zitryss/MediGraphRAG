"""Local filesystem side effect for the educational teapot endpoint."""

import os
from pathlib import Path
import secrets
import tempfile
import uuid


def create_teapot_file(project_root: Path) -> str:
    contents = "".join(secrets.choice("0123456789") for _ in range(32)) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=project_root,
        prefix=".teapot-", suffix=".tmp", delete=False,
    ) as staging:
        staging_path = Path(staging.name)
        try:
            staging.write(contents)
            staging.flush()
            staging.close()
            for _ in range(10):
                filename = f"teapot-{uuid.uuid4().hex}.txt"
                try:
                    # Publish complete bytes atomically; unlike rename, link cannot
                    # replace an existing file. Both paths are on the same filesystem.
                    os.link(staging_path, project_root / filename)
                except FileExistsError:
                    continue
                return filename
            raise FileExistsError("Could not allocate a unique teapot filename")
        finally:
            staging_path.unlink(missing_ok=True)
