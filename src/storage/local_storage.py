"""
Local File Storage Manager.
Handles saving, retrieving, and safely cleaning up resume PDF files on the local filesystem.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_storage_base_dir() -> Path:
    """
    Returns the absolute path to the local resume storage directory.
    Default: <project_root>/data/resumes
    """
    custom_dir = os.getenv("RESUME_STORAGE_DIR")
    if custom_dir:
        base = Path(custom_dir)
    else:
        # Default to data/resumes under project root
        base = Path(__file__).resolve().parent.parent.parent / "data" / "resumes"

    base.mkdir(parents=True, exist_ok=True)
    return base


def save_resume_file(file_bytes: bytes, relative_path: str) -> str:
    """
    Saves PDF file bytes to the local filesystem at the given relative path.

    :param file_bytes: Binary PDF content.
    :param relative_path: Safe path such as '<uuid>/<filename>.pdf'.
    :return: Absolute file path as string.
    """
    base_dir = get_storage_base_dir()
    target_path = (base_dir / relative_path).resolve()

    # Prevent directory traversal outside base_dir
    if not str(target_path).startswith(str(base_dir.resolve())):
        raise ValueError(f"Security error: Storage path traversal detected: {relative_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(file_bytes)

    logger.info("Saved resume PDF to local storage: %s", target_path)
    # Store relative or standard storage path
    return str(target_path)


def delete_resume_file(relative_or_abs_path: str) -> bool:
    """
    Safely deletes a resume file from local storage.

    :param relative_or_abs_path: Relative storage path or absolute path.
    :return: True if deleted or already absent, False if error.
    """
    try:
        base_dir = get_storage_base_dir()
        path_obj = Path(relative_or_abs_path)

        if not path_obj.is_absolute():
            path_obj = base_dir / relative_or_abs_path

        if path_obj.exists() and path_obj.is_file():
            path_obj.unlink()
            logger.info("Deleted resume file from local storage: %s", path_obj)
            # Try removing empty parent directory if inside base_dir
            try:
                parent = path_obj.parent
                if parent != base_dir and parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception:
                pass
        return True
    except Exception as e:
        logger.error("Failed to delete local resume file '%s': %s", relative_or_abs_path, str(e))
        return False


def get_resume_file_path(relative_path: str) -> Optional[Path]:
    """
    Retrieves the absolute path of a stored resume file if it exists.
    """
    base_dir = get_storage_base_dir()
    target = (base_dir / relative_path).resolve()
    if target.exists() and target.is_file():
        return target
    return None
