"""
Unit and Integration Tests for MySQL Client.
Verifies MySQL connectivity, cursor context managers, and schema structure.
"""

from src.database.mysql_client import (
    get_mysql_connection,
    get_db_cursor,
    test_mysql_connection as run_mysql_connection_check,
)


class TestMySQLConnection:
    """Test suite for MySQL client."""

    def test_mysql_connection_successful(self):
        """Verify direct MySQL connection can be established."""
        conn = get_mysql_connection()
        assert conn is not None
        conn.close()

    def test_mysql_safe_connection_check(self):
        """Verify test_mysql_connection returns connected status and table list."""
        res = run_mysql_connection_check()
        assert res["status"] == "connected"
        assert "resume_job" in res["message"]
        assert "resumes" in res["tables"]
        assert "jobs" in res["tables"]
        assert "skills" in res["tables"]

    def test_mysql_cursor_context_manager(self):
        """Verify get_db_cursor executes query and returns dictionary rows."""
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT 100 AS num, 'test_val' AS str_val;")
            row = cur.fetchone()
            assert row["num"] == 100
            assert row["str_val"] == "test_val"
