"""
MySQL Database Client Module.
Provides connection pooling, connection management, schema initialization,
and safe query execution for MySQL database 'resume_job'.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dotenv import load_dotenv
import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)


def get_mysql_config() -> Dict[str, Any]:
    """
    Load and return MySQL configuration from environment variables.
    """
    load_dotenv()
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "resume_job"),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": True,
    }


def get_mysql_connection() -> pymysql.Connection:
    """
    Creates and returns a new MySQL database connection.
    """
    config = get_mysql_config()
    try:
        connection = pymysql.connect(**config)
        return connection
    except Exception as e:
        logger.error("Failed to connect to MySQL database '%s': %s", config["database"], str(e))
        raise RuntimeError(f"Could not connect to MySQL database: {str(e)}") from e


@contextmanager
def get_db_cursor(commit: bool = True):
    """
    Context manager for database operations with automatic commit and rollback.
    """
    conn = get_mysql_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def test_mysql_connection() -> Dict[str, Any]:
    """
    Performs a safe read-only connectivity check against MySQL.
    """
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT 1 AS status_check;")
            result = cur.fetchone()
            cur.execute("SHOW TABLES;")
            tables = [list(row.values())[0] for row in cur.fetchall()]

        return {
            "status": "connected",
            "message": "Successfully connected to MySQL database 'resume_job'.",
            "tables": tables,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"MySQL connection failed: {str(e)}",
        }


def init_mysql_schema(schema_file: Optional[Path] = None) -> bool:
    """
    Executes the MySQL schema definition to ensure all required tables exist.
    """
    if schema_file is None:
        schema_file = Path(__file__).resolve().parent.parent.parent / "data" / "mysql_schema.sql"

    if not schema_file.exists():
        logger.error("Schema file not found at: %s", schema_file)
        return False

    with open(schema_file, "r", encoding="utf-8") as f:
        sql_script = f.read()

    statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]

    try:
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()
        conn.close()
        logger.info("MySQL schema initialized successfully.")
        return True
    except Exception as e:
        logger.error("Failed to initialize MySQL schema: %s", str(e))
        return False


def seed_skills_taxonomy(taxonomy_file: Optional[Path] = None) -> int:
    """
    Seeds the MySQL 'skills' table with the 100-skill master taxonomy from CSV.
    Uses INSERT IGNORE or ON DUPLICATE KEY UPDATE to ensure idempotency.
    :return: Count of seeded skills.
    """
    import csv

    if taxonomy_file is None:
        taxonomy_file = Path(__file__).resolve().parent.parent.parent / "data" / "taxonomy" / "skills.csv"

    if not taxonomy_file.exists():
        logger.warning("Taxonomy CSV file not found at: %s", taxonomy_file)
        return 0

    skills_data = []
    with open(taxonomy_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills_data.append((
                row["skill_id"].strip(),
                row["skill_name"].strip(),
                row["category"].strip(),
                row.get("aliases", "").strip()
            ))

    if not skills_data:
        return 0

    query = """
        INSERT INTO skills (id, skill_name, category, aliases)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            category = VALUES(category),
            aliases = VALUES(aliases);
    """

    with get_db_cursor(commit=True) as cur:
        cur.executemany(query, skills_data)

    logger.info("Successfully seeded %d skills into MySQL skills table.", len(skills_data))
    return len(skills_data)


def seed_benchmark_jobs(jobs_file: Optional[Path] = None) -> int:
    """
    Seeds the MySQL 'jobs' and 'job_skills' tables with the benchmark jobs from CSV.
    Uses ON DUPLICATE KEY UPDATE to ensure idempotency.
    :return: Count of seeded jobs.
    """
    import csv

    if jobs_file is None:
        jobs_file = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "jobs" / "jobs.csv"

    if not jobs_file.exists():
        logger.warning("Jobs CSV file not found at: %s", jobs_file)
        return 0

    jobs_data = []
    job_skills_data = []

    # Map skill_name -> skill_id from DB
    skill_map = {}
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT id, skill_name FROM skills")
            for r in cur.fetchall():
                skill_map[r["skill_name"].lower()] = r["id"]
    except Exception as e:
        logger.warning("Could not load skills for job_skills seeding: %s", str(e))

    with open(jobs_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = row["job_id"].strip()
            min_exp = float(row.get("minimum_experience", 0.0) or 0.0)
            jobs_data.append((
                job_id,
                row["job_title"].strip(),
                row["company"].strip(),
                row["job_description"].strip(),
                row["job_category"].strip(),
                min_exp,
                row.get("education_requirement", "").strip()
            ))

            # Parse required and preferred skills
            req_skills = [s.strip() for s in row.get("required_skills", "").split(";") if s.strip()]
            for sk in req_skills:
                sk_id = skill_map.get(sk.lower())
                if sk_id:
                    job_skills_data.append((job_id, sk_id, True))

            pref_skills = [s.strip() for s in row.get("preferred_skills", "").split(";") if s.strip()]
            for sk in pref_skills:
                sk_id = skill_map.get(sk.lower())
                if sk_id:
                    job_skills_data.append((job_id, sk_id, False))

    if not jobs_data:
        return 0

    query_jobs = """
        INSERT INTO jobs (id, job_title, company, job_description, job_category, minimum_experience, education_requirement)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            job_title = VALUES(job_title),
            company = VALUES(company),
            job_description = VALUES(job_description),
            job_category = VALUES(job_category),
            minimum_experience = VALUES(minimum_experience),
            education_requirement = VALUES(education_requirement);
    """

    query_job_skills = """
        INSERT INTO job_skills (job_id, skill_id, is_required)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            is_required = VALUES(is_required);
    """

    with get_db_cursor(commit=True) as cur:
        cur.executemany(query_jobs, jobs_data)
        if job_skills_data:
            cur.executemany(query_job_skills, job_skills_data)

    logger.info("Successfully seeded %d benchmark jobs into MySQL.", len(jobs_data))
    return len(jobs_data)


def seed_benchmark_resumes(resumes_file: Optional[Path] = None) -> int:
    """
    Seeds the MySQL 'resumes' and 'resume_skills' tables with the candidate resumes from CSV.
    Uses ON DUPLICATE KEY UPDATE to ensure idempotency.
    :return: Count of seeded resumes.
    """
    import csv

    if resumes_file is None:
        resumes_file = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "resumes" / "resumes.csv"

    if not resumes_file.exists():
        logger.warning("Resumes CSV file not found at: %s", resumes_file)
        return 0

    resumes_data = []
    resume_skills_data = []

    # Map skill_name -> skill_id from DB
    skill_map = {}
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT id, skill_name FROM skills")
            for r in cur.fetchall():
                skill_map[r["skill_name"].lower()] = r["id"]
    except Exception as e:
        logger.warning("Could not load skills for resume_skills seeding: %s", str(e))

    with open(resumes_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res_id = row["resume_id"].strip()
            exp_years = float(row.get("experience_years", 0.0) or 0.0)
            resumes_data.append((
                res_id,
                f"{res_id}.pdf",
                f"data/raw/resumes/{res_id}.pdf",
                row["resume_text"].strip(),
                row.get("education", "").strip(),
                row.get("degree", "").strip(),
                row.get("field_of_study", "").strip(),
                exp_years
            ))

            raw_skills = [s.strip() for s in row.get("skills", "").split(";") if s.strip()]
            for sk in raw_skills:
                sk_id = skill_map.get(sk.lower())
                if sk_id:
                    resume_skills_data.append((res_id, sk_id))

    if not resumes_data:
        return 0

    query_resumes = """
        INSERT INTO resumes (id, filename, file_path, extracted_text, education, degree, field_of_study, experience_years)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            filename = VALUES(filename),
            file_path = VALUES(file_path),
            extracted_text = VALUES(extracted_text),
            education = VALUES(education),
            degree = VALUES(degree),
            field_of_study = VALUES(field_of_study),
            experience_years = VALUES(experience_years);
    """

    query_resume_skills = """
        INSERT INTO resume_skills (resume_id, skill_id)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
            skill_id = VALUES(skill_id);
    """

    with get_db_cursor(commit=True) as cur:
        cur.executemany(query_resumes, resumes_data)
        if resume_skills_data:
            cur.executemany(query_resume_skills, resume_skills_data)

    logger.info("Successfully seeded %d candidate resumes into MySQL.", len(resumes_data))
    return len(resumes_data)


