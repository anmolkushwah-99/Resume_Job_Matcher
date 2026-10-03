"""
Local file storage module for candidate resumes.
"""

from src.storage.local_storage import (
    save_resume_file,
    delete_resume_file,
    get_resume_file_path,
    get_storage_base_dir,
)

__all__ = [
    "save_resume_file",
    "delete_resume_file",
    "get_resume_file_path",
    "get_storage_base_dir",
]
