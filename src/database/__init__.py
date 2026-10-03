"""
Database integration module.
Provides connection and client helpers for MySQL database 'resume_job'.
"""

from src.database.mysql_client import (
    get_mysql_connection,
    get_db_cursor,
    test_mysql_connection,
    init_mysql_schema,
    seed_skills_taxonomy,
)

__all__ = [
    "get_mysql_connection",
    "get_db_cursor",
    "test_mysql_connection",
    "init_mysql_schema",
    "seed_skills_taxonomy",
]
