import hashlib
from pathlib import Path


def compute_file_hash(
    file_path: str | Path,
) -> str:
    """
    Compute SHA256 hash of a file.
    """

    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)

    return sha.hexdigest()